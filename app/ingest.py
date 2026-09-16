"""
Ingestion pipeline for the RAG system.

Flow:
  1. Load raw documents (txt / pdf) from a folder
  2. Add stable document metadata
  3. Split documents into overlapping chunks
  4. Embed chunks using a local HuggingFace model
  5. Store embeddings in FAISS
"""

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents(source_dir: str):
    docs = []

    for fname in os.listdir(source_dir):
        path = os.path.join(source_dir, fname)

        if not os.path.isfile(path):
            continue

        document_name = fname

        if fname.lower().endswith(".pdf"):
            loaded_docs = PyPDFLoader(path).load()

        elif fname.lower().endswith(".txt"):
            loaded_docs = TextLoader(path, encoding="utf-8").load()

        else:
            continue

        # Store the document name in every page/chunk's metadata.
        for doc in loaded_docs:
            doc.metadata["document"] = document_name
            doc.metadata["source"] = document_name

        docs.extend(loaded_docs)

    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def build_vector_store(chunks, persist_dir: str = VECTOR_STORE_DIR):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(persist_dir)

    return vector_store


def run_ingestion(source_dir: str = "data/sample_docs"):
    print(f"[1/3] Loading documents from '{source_dir}'...")

    documents = load_documents(source_dir)

    print(f"      Loaded {len(documents)} document pages." )

    print("[2/3] Chunking documents...")

    chunks = chunk_documents(documents)

    print(
        f"      Created {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    print("[3/3] Embedding chunks and building FAISS index...")

    build_vector_store(chunks)

    print(f"      Vector store saved to '{VECTOR_STORE_DIR}/'")


if __name__ == "__main__":
    run_ingestion()