"""
FastAPI entrypoint for the DocQA RAG service.

Run:  python -m uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil

from app.graph import ask as rag_ask
from app.ingest import run_ingestion


app = FastAPI(
    title="DocQA - RAG based Document Q&A API",
    description=(
        "Ask questions over your own documents using RAG "
        "(FastAPI + LangChain + LangGraph + FAISS), "
        "with conversation memory."
    ),
    version="1.1.0",
)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    document: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    used_retrieval: bool
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def get_documents():
    """Return all uploaded PDF/TXT documents."""

    source_dir = "data/sample_docs"

    if not os.path.exists(source_dir):
        return {"documents": []}

    documents = [
        fname
        for fname in os.listdir(source_dir)
        if fname.lower().endswith((".pdf", ".txt"))
        and os.path.isfile(os.path.join(source_dir, fname))
    ]

    return {
        "documents": sorted(documents)
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    result = rag_ask(
        payload.question,
        session_id=payload.session_id,
        document=payload.document,
    )

    return AnswerResponse(**result)


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Upload a PDF/TXT and refresh the FAISS index."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    os.makedirs("data/sample_docs", exist_ok=True)

    dest = os.path.join(
        "data/sample_docs",
        file.filename,
    )

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_ingestion()

    return {
        "message": f"'{file.filename}' ingested and index refreshed."
    }