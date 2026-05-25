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
TOP_K = 8
MAX_CONTEXT_CHARS = 3000
MAX_FILE_SIZE_MB = 10

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment variables.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


FACT_PROMPT = """
Answer the question using ONLY the context below.
Be precise and factual.

If the answer is not present in the context, say:
"I cannot find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

INSIGHT_PROMPT = """
You are analyzing real content provided by the user.

Use the context below and apply logical reasoning.
You may infer and connect ideas, but do not invent numbers or facts.
If you make an assumption, clearly state it.

Write naturally and clearly.

Context:
{context}

Question:
{question}

Answer:
"""


def is_allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def detect_intent(question):
    q = question.lower()

    insight_keywords = [
        "why", "suggest", "recommend", "improve", "optimize",
        "strategy", "risk", "insight", "analysis", "opinion",
        "should", "better", "focus", "advise"
    ]

    for word in insight_keywords:
        if word in q:
            return "insight"

    return "fact"


def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if len(chunk) >= 100:
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
        "User-Agent": "Mozilla/5.0 (compatible; SmartRAGBot/1.0)"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError("This link does not appear to contain readable webpage text.")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())

    return text


def save_index(index, documents):
    faiss.write_index(index, os.path.join(INDEX_FOLDER, "index.faiss"))

    with open(os.path.join(INDEX_FOLDER, "docs.pkl"), "wb") as file:
        pickle.dump(documents, file)


def load_index():
    index_path = os.path.join(INDEX_FOLDER, "index.faiss")
    docs_path = os.path.join(INDEX_FOLDER, "docs.pkl")

    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return None, None

    index = faiss.read_index(index_path)

    with open(docs_path, "rb") as file:
        documents = pickle.load(file)

    return index, documents


def build_index_from_text(raw_text):
    documents = chunk_text(raw_text)

    if not documents:
        raise ValueError("The content is too small or could not be chunked properly.")

    embeddings = embedder.encode(documents, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    save_index(index, documents)


@app.route("/", methods=["GET"])
def home():
    uploaded = request.args.get("uploaded") == "1"
    indexed = os.path.exists(os.path.join(INDEX_FOLDER, "index.faiss"))

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

    if not filename or not is_allowed_file(filename):
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error="Only TXT, PDF, and CSV files are supported."
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
            return render_template(
                "index.html",
                indexed=False,
                uploaded=False,
                answer=None,
                error="No readable text was found in the document."
            )

        build_index_from_text(raw_text)

        return redirect(url_for("home", uploaded=1))

    except Exception as error:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error=f"Failed to process document: {str(error)}"
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
            error="Please enter a valid http or https URL."
        )

    try:
        raw_text = load_url(url)

        if not raw_text.strip():
            return render_template(
                "index.html",
                indexed=False,
                uploaded=False,
                answer=None,
                error="No readable text was found at this link."
            )

        build_index_from_text(raw_text)

        return redirect(url_for("home", uploaded=1))

    except Exception as error:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error=f"Failed to read link: {str(error)}"
        )


@app.route("/ask", methods=["POST"])
def ask():
    index, documents = load_index()

    if index is None or documents is None:
        return render_template(
            "index.html",
            indexed=False,
            uploaded=False,
            answer=None,
            error="Please upload a document or index a link first."
        )

    query = request.form.get("query", "").strip()

    if not query:
        return redirect(url_for("home"))

    try:
        query_embedding = embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")

        _, indexes = index.search(query_embedding, k=min(TOP_K, len(documents)))

        context_chunks = []
        total_chars = 0

        for doc_index in indexes[0]:
            if doc_index < 0 or doc_index >= len(documents):
                continue

            chunk = documents[doc_index]

            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break

            context_chunks.append(chunk)
            total_chars += len(chunk)

        context = "\n\n".join(context_chunks)

        if not context:
            context = documents[0][:MAX_CONTEXT_CHARS]

        intent = detect_intent(query)

        if intent == "insight":
            prompt = INSIGHT_PROMPT.format(context=context, question=query)
            temperature = 0.4
        else:
            prompt = FACT_PROMPT.format(context=context, question=query)
            temperature = 0.2

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=600
        )

        answer = response.choices[0].message.content

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
            error=f"Failed to generate answer: {str(error)}"
        )


@app.route("/reset", methods=["POST"])
def reset():
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
        error=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
