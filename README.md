# Smart RAG Document Q&A

A Flask-based Retrieval-Augmented Generation application that allows users to upload documents and ask natural-language questions about their content.

## Features

- Upload TXT, PDF, and CSV files
- Extract and chunk document text
- Generate semantic embeddings using SentenceTransformers
- Store and search document chunks using FAISS
- Generate grounded answers using Groq's OpenAI-compatible API
- Clean Flask web interface
- Secure file upload handling
- Production-ready deployment using Gunicorn

## Tech Stack

- Python
- Flask
- FAISS
- SentenceTransformers
- Groq API
- PyPDF2
- NumPy
- Gunicorn

## Setup

```bash
git clone your-repo-url
cd data_rag_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
