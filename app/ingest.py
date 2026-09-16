"""
Ingestion pipeline for the RAG system.

Flow:
  1. Load raw documents (txt / pdf) from a folder
  2. Split them into overlapping chunks (RecursiveCharacterTextSplitter)
  3. Embed each chunk using a local HuggingFace sentence-transformer
  4. Store the embeddings in a FAISS vector index and persist it to disk

Run:  python -m app.ingest
"""
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(source_dir: str):
    docs = []
    for fname in os.listdir(source_dir):
        path = os.path.join(source_dir, fname)
        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif fname.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, persist_dir: str = VECTOR_STORE_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(persist_dir)
    return vector_store


def run_ingestion(source_dir: str = "data/sample_docs"):
    print(f"[1/3] Loading documents from '{source_dir}'...")
    documents = load_documents(source_dir)
    print(f"      Loaded {len(documents)} document(s).")

    print("[2/3] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"      Created {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("[3/3] Embedding chunks and building FAISS index...")
    build_vector_store(chunks)
    print(f"      Vector store saved to '{VECTOR_STORE_DIR}/'")


if __name__ == "__main__":
    run_ingestion()
