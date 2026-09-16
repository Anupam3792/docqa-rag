"""
FastAPI entrypoint for the DocQA RAG service.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os, shutil

from app.graph import rag_graph
from app.ingest import run_ingestion

app = FastAPI(
    title="DocQA - RAG based Document Q&A API",
    description="Ask questions over your own documents using RAG (FastAPI + LangChain + LangGraph + FAISS).",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    used_retrieval: bool
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_graph.invoke({"question": payload.question})
    return AnswerResponse(
        answer=result["answer"],
        used_retrieval=result["needs_retrieval"],
        sources=[c["source"] for c in result.get("chunks", [])],
    )


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Upload a new .txt or .pdf, then re-run ingestion to refresh the index."""
    os.makedirs("data/sample_docs", exist_ok=True)
    dest = os.path.join("data/sample_docs", file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_ingestion()
    return {"message": f"'{file.filename}' ingested and index refreshed."}
