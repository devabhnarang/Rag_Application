from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import csv
import os
import pickle
import tempfile

import faiss
import numpy as np
import PyPDF2
import requests

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "data_rag_uploads")
INDEX_FOLDER = os.path.join(tempfile.gettempdir(), "data_rag_store")

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv"}

TOP_K = 4
MAX_CONTEXT_CHARS = 1800
MAX_FILE_SIZE_MB = 10

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

embedder = None
client = None

chat_history = []

MAIN_PROMPT = """
You are a smart and helpful AI assistant.

Use the provided context to answer the user's question.

Rules:
- Prefer information from the context.
- If the answer is partially available, answer with available details.
- If the answer is obvious from the context, infer carefully.
- Explain naturally like ChatGPT.
- DO NOT unnecessarily say information is missing.
- Only say information is unavailable if absolutely nothing relevant exists.
- Keep answers clean and helpful.

Context:
{context}

Question:
{question}

Helpful Answer:
"""


def get_embedder():
    global embedder

    if embedder is None:
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    return embedder


def get_groq_client():
    global client

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    return client


def is_allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def chunk_text(text, chunk_size=220, overlap=40):
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()

        if len(chunk) >= 80:
            chunks.append(chunk)

    return chunks


def load_csv(path):
    text = ""

    with open(path, encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)

        for row in reader:
            text += " ".join(row) + " "

    return text


def load_pdf(path):
    text = ""

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text


def load_txt(path):
    with open(path, encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=15)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "form"
    ]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    text = " ".join(text.split())

    return text


def save_index(index, documents):
    faiss.write_index(
        index,
        os.path.join(INDEX_FOLDER, "index.faiss")
    )

    with open(
        os.path.join(INDEX_FOLDER, "docs.pkl"),
        "wb"
    ) as file:
        pickle.dump(documents, file)


def load_index():
    index_path = os.path.join(INDEX_FOLDER, "index.faiss")
    docs_path = os.path.join(INDEX_FOLDER, "docs.pkl")

    if not os.path.exists(index_path):
        return None, None

    if not os.path.exists(docs_path):
        return None, None

    index = faiss.read_index(index_path)

    with open(docs_path, "rb") as file:
        documents = pickle.load(file)

    return index, documents


def build_index_from_text(raw_text):
    documents = chunk_text(raw_text)

    if not documents:
        raise ValueError("Document content is too small.")

    embeddings = get_embedder().encode(
        documents,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    save_index(index, documents)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/", methods=["GET"])
def home():
    uploaded = request.args.get("uploaded") == "1"

    indexed = os.path.exists(
        os.path.join(INDEX_FOLDER, "index.faiss")
    )

    return render_template(
        "index.html",
        indexed=indexed,
        uploaded=uploaded,
        answer=None,
        error=None
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or not file.filename:
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)

    ext = os.path.splitext(filename)[1].lower()

    if not is_allowed_file(filename):
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error="Only TXT, PDF and CSV supported."
        )

    path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(path)

    try:
        if ext == ".csv":
            raw_text = load_csv(path)

        elif ext == ".pdf":
            raw_text = load_pdf(path)

        else:
            raw_text = load_txt(path)

        if not raw_text.strip():
            raise ValueError("No readable text found.")

        build_index_from_text(raw_text)

        return redirect(url_for("home", uploaded=1))

    except Exception as error:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error=str(error)
        )


@app.route("/link", methods=["POST"])
def link():
    url = request.form.get("url", "").strip()

    if not url:
        return redirect(url_for("home"))

    if not is_valid_url(url):
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error="Enter valid URL."
        )

    try:
        raw_text = load_url(url)

        if not raw_text.strip():
            raise ValueError("No readable text found.")

        build_index_from_text(raw_text)

        return redirect(url_for("home", uploaded=1))

    except Exception as error:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error=str(error)
        )


@app.route("/ask", methods=["POST"])
def ask():
    global chat_history

    index, documents = load_index()

    if index is None or documents is None:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error="Upload document first."
        )

    query = request.form.get("query", "").strip()

    if not query:
        return redirect(url_for("home"))

    try:
        query_embedding = get_embedder().encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.array(query_embedding).astype("float32")

        scores, indexes = index.search(
            query_embedding,
            k=min(TOP_K, len(documents))
        )

        context_chunks = []

        total_chars = 0

        for score, doc_index in zip(scores[0], indexes[0]):

            if score < 0.35:
                continue

            if doc_index < 0 or doc_index >= len(documents):
                continue

            chunk = documents[doc_index]

            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break

            context_chunks.append(chunk)

            total_chars += len(chunk)

        if not context_chunks:
            context_chunks.append(documents[0][:800])

        context = "\n\n".join(context_chunks)

        prompt = MAIN_PROMPT.format(
            context=context,
            question=query
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful RAG AI assistant."
            }
        ]

        messages.extend(chat_history[-6:])

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )

        answer = response.choices[0].message.content

        chat_history.append({
            "role": "user",
            "content": query
        })

        chat_history.append({
            "role": "assistant",
            "content": answer
        })

        return render_template(
            "index.html",
            indexed=True,
            uploaded=False,
            answer=answer,
            error=None
        )

    except Exception as error:
        return render_template(
            "index.html",
            indexed=True,
            uploaded=False,
            answer=None,
            error=f"Failed: {str(error)}"
        )


@app.route("/reset", methods=["POST"])
def reset():
    global chat_history

    chat_history = []

    for filename in ["index.faiss", "docs.pkl"]:

        path = os.path.join(INDEX_FOLDER, filename)

        if os.path.exists(path):
            os.remove(path)

    for filename in os.listdir(UPLOAD_FOLDER):

        path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.isfile(path):
            os.remove(path)

    return redirect(url_for("home"))


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        indexed=False,
        uploaded=False,
        answer=None,
        error=f"File too large. Max size {MAX_FILE_SIZE_MB}MB."
    ), 413


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
