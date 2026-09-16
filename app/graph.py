"""
LangGraph orchestration layer.

RAG flow:
START -> classify_query -> retrieve -> generate -> END
                         -> generate_direct -> END
"""

from typing import TypedDict, List
import requests

from langgraph.graph import StateGraph, END

from app.vectorstore import retrieve as vector_retrieve


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"


class GraphState(TypedDict):
    question: str
    needs_retrieval: bool
    chunks: List[dict]
    answer: str


def classify_query(state: GraphState) -> GraphState:
    """Route normal questions through RAG and trivial questions directly."""
    trivial = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}
    q = state["question"].strip().lower()
    state["needs_retrieval"] = q not in trivial and len(q.split()) > 2
    return state


def route(state: GraphState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate_direct"


def retrieve_node(state: GraphState) -> GraphState:
    state["chunks"] = vector_retrieve(state["question"])
    return state


def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(c["content"] for c in state["chunks"])

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n"
        "Answer:"
    )

    state["answer"] = _call_llm(prompt)
    return state


def generate_direct_node(state: GraphState) -> GraphState:
    state["chunks"] = []
    state["answer"] = _call_llm(state["question"])
    return state


def _call_llm(prompt: str) -> str:
    """Generate an answer using the local Ollama model."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        return response.json()["response"]

    except requests.exceptions.RequestException as e:
        return f"[Ollama error: {e}]"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_query", classify_query)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("generate_direct", generate_direct_node)

    workflow.set_entry_point("classify_query")

    workflow.add_conditional_edges(
        "classify_query",
        route,
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct",
        },
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_direct", END)

    return workflow.compile()


rag_graph = build_graph()