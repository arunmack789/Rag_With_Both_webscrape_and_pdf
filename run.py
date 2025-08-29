import gradio as gr
import fitz
import numpy as np
import faiss
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# === PDF Extraction ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        print(f"Page {page.number}: {len(page_text)} characters")  # Debug
        text += page_text + "\n"
    print(f"Total extracted text length: {len(text)}")  # Debug
    return text

# === Web scraping ===
def scrape_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print(f"Fetched {len(response.text)} characters from {url}")  # Debug
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=" ")
        cleaned = " ".join(text.split())
        print(f"Extracted {len(cleaned)} characters of cleaned text")  # Debug
        return cleaned
    except Exception as e:
        print(f"Scraping error: {e}")  # Debug
        return f"Failed to scrape {url}: {e}"

# === Chunking ===
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# === Embeddings ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_chunks(chunks):
    return embed_model.encode(chunks, convert_to_numpy=True)

# === FAISS Index ===
def build_faiss_index(chunks, embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# === Retrieve relevant chunks ===
def retrieve(query, index, chunks, top_k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

# === Llama streaming ===
llm = Llama(
    model_path="Llama-3.2-3B-I-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
)

def stream_llama(prompt, max_tokens=512, temperature=0):
    text_so_far = ""
    for token in llm(prompt, max_tokens=max_tokens, stream=True, temperature=temperature):
        new_text = token["choices"][0]["text"]
        text_so_far += new_text
        yield text_so_far

# === Chat function with streaming ===
def chat_with_pdf_web_stream(message, chat_history, pdf_file, url, session_data):
    if chat_history is None or len(chat_history) == 0:
        chat_history = [{"role": "system", "content": "Hello! I am Doozy, your AI assistant..."}]

    # Handle greetings
    if message.strip().lower() in {"hi", "hello", "hey", "Hi"}:
        chat_history.append({"role": "assistant", "content": "Hi there! Ask me something about your PDF or website."})
        yield chat_history, chat_history
        return

    # === PDF Handling ===
    if pdf_file is None and session_data.get("pdf_index") is not None:
        session_data["pdf_index"] = None
        session_data["pdf_chunks"] = None
        chat_history.append({"role": "system", "content": "PDF removed. Context cleared."})

    if pdf_file is not None and session_data.get("pdf_index") is None:
        text = extract_text_from_pdf(pdf_file.name)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(chunks, embeddings)
        session_data["pdf_index"] = index
        session_data["pdf_chunks"] = chunks
        chat_history.append({"role": "system", "content": "Doozy has loaded your PDF."})

    # === Web Handling ===
    if url:
        if session_data.get("web_url") != url:
            session_data["web_index"] = None
            session_data["web_chunks"] = None
            session_data["web_url"] = url
            chat_history.append({"role": "system", "content": f"New URL detected. Reloading: {url}"})

        if session_data.get("web_index") is None:
            web_text = scrape_webpage(url)
            web_chunks = chunk_text(web_text)
            web_embeddings = embed_chunks(web_chunks)
            web_index = build_faiss_index(web_chunks, web_embeddings)
            session_data["web_index"] = web_index
            session_data["web_chunks"] = web_chunks
            chat_history.append({"role": "system", "content": f"Doozy has loaded the web page: {url}"})

    chat_history.append({"role": "user", "content": message})

    pdf_relevant = retrieve(message, session_data["pdf_index"], session_data["pdf_chunks"]) if session_data.get("pdf_index") else []
    web_relevant = retrieve(message, session_data["web_index"], session_data["web_chunks"]) if session_data.get("web_index") else []
    combined_context = " ".join(pdf_relevant + web_relevant)

    prompt = (
        f"You are Doozy, an AI assistant. Answer the question professionally using the context below.\n\n"
        f"Context:\n{combined_context}\n\nUser Question: {message}\nAnswer as Doozy:"
    )

    answer = ""
    for partial in stream_llama(prompt):
        answer = partial
        if len(chat_history) == 0 or chat_history[-1]["role"] != "assistant":
            chat_history.append({"role": "assistant", "content": answer})
        else:
            chat_history[-1]["content"] = answer
        yield chat_history, chat_history

# === Save chat history ===
def save_chat_history(chat_history, filename="chat_history.json"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{filename.rstrip('.json')}_{timestamp}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4, ensure_ascii=False)
    return f"Chat history saved to {save_path}"

# === Gradio Interface ===
with gr.Blocks() as demo:
    pdf_file = gr.File(file_types=[".pdf"], label="Upload PDF (Optional)")
    url_input = gr.Textbox(placeholder="Enter website URL to scrape (Optional)", label="Website URL")
    chatbot = gr.Chatbot(label="Doozy Chatbot", type="messages")
    msg = gr.Textbox(placeholder="Ask a question...", label="Message")
    clear = gr.Button("Clear Chat")
    save_btn = gr.Button("Save Chat History")
    save_status = gr.Textbox(label="Save Status")
    clear_sources = gr.Button("Clear PDF & Web Context")

    session_data = {"pdf_index": None, "pdf_chunks": None, "web_index": None, "web_chunks": None, "web_url": None}

    msg.submit(chat_with_pdf_web_stream, [msg, chatbot, pdf_file, url_input, gr.State(session_data)], [chatbot, chatbot])

    clear.click(lambda: [
        {"role": "system", "content": "Hello! I am Doozy, your AI assistant. I can help you with questions using PDFs, web content, or general knowledge. How can I assist you today?"}
    ], None, chatbot)

    save_btn.click(save_chat_history, [chatbot], [save_status])

    clear_sources.click(lambda: {
        "pdf_index": None,
        "pdf_chunks": None,
        "web_index": None,
        "web_chunks": None,
        "web_url": None
    }, None, gr.State(session_data))

demo.launch()
