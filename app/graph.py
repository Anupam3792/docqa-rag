"""
LangGraph orchestration layer with:
- conversation memory
- document-aware retrieval
- direct-answer routing for small talk
"""

import operator
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import ollama

from app.config import OLLAMA_MODEL, OLLAMA_HOST
from app.vectorstore import retrieve as vector_retrieve


MAX_HISTORY_TURNS = 5

_ollama_client = ollama.Client(host=OLLAMA_HOST)


class GraphState(TypedDict, total=False):
    question: str
    document: str | None
    needs_retrieval: bool
    chunks: List[dict]
    answer: str
    history: Annotated[List[dict], operator.add]


def classify_query(state: GraphState) -> GraphState:
    """Route greetings/small-talk without vector retrieval."""

    trivial = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
    }

    q = state["question"].strip().lower()
    state["needs_retrieval"] = q not in trivial and len(q.split()) > 2

    return state


def route(state: GraphState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate_direct"


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve chunks only from the selected document when provided."""

    state["chunks"] = vector_retrieve(
        state["question"],
        document=state.get("document"),
    )

    return state


def _format_history(state: GraphState) -> str:
    past_turns = state.get("history", [])[-MAX_HISTORY_TURNS:]

    if not past_turns:
        return ""

    lines = [
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in past_turns
    ]

    return (
        "Previous conversation:\n"
        + "\n\n".join(lines)
        + "\n\n"
    )


def generate_node(state: GraphState) -> dict:
    context = "\n\n".join(
        chunk["content"] for chunk in state.get("chunks", [])
    )

    history_block = _format_history(state)

    document_name = state.get("document")

    if document_name:
        scope_instruction = (
            f"You are answering questions about the document "
            f"'{document_name}'."
        )
    else:
        scope_instruction = (
            "You are answering questions using the available documents."
        )

    prompt = (
        f"{scope_instruction}\n\n"
        "Answer the question using ONLY the document context below "
        "and previous conversation for follow-up context.\n"
        "If the answer is not present in the context, say you don't know. "
        "Do not invent information.\n\n"
        f"{history_block}"
        f"Document context:\n{context}\n\n"
        f"Question: {state['question']}\n"
        "Answer:"
    )

    answer = _call_llm(prompt)

    return {
        "answer": answer,
        "history": [
            {
                "question": state["question"],
                "answer": answer,
            }
        ],
    }


def generate_direct_node(state: GraphState) -> dict:
    history_block = _format_history(state)

    prompt = (
        f"{history_block}"
        f"Question: {state['question']}\n"
        "Answer:"
    )

    answer = _call_llm(prompt)

    return {
        "chunks": [],
        "answer": answer,
        "history": [
            {
                "question": state["question"],
                "answer": answer,
            }
        ],
    }


def _call_llm(prompt: str) -> str:
    """Call the local Ollama model."""

    try:
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0.2,
            },
        )

        return response["message"]["content"]

    except Exception as e:
        return (
            f"[Could not reach Ollama at {OLLAMA_HOST} — "
            f"is Ollama running and is the '{OLLAMA_MODEL}' model pulled? "
            f"Error: {e}]"
        )


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(
        "classify_query",
        classify_query,
    )

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "generate",
        generate_node,
    )

    workflow.add_node(
        "generate_direct",
        generate_direct_node,
    )

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

    memory = MemorySaver()

    return workflow.compile(
        checkpointer=memory
    )


rag_graph = build_graph()


def ask(
    question: str,
    session_id: str = "default",
    document: str | None = None,
) -> dict:
    """
    Run one RAG turn.

    document:
      - filename -> search only that document
      - None -> search across all documents
    """

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result = rag_graph.invoke(
        {
            "question": question,
            "document": document,
        },
        config=config,
    )

    # Remove duplicate source names.
    sources = list(
        dict.fromkeys(
            chunk["source"]
            for chunk in result.get("chunks", [])
            if chunk.get("source")
        )
    )

    return {
        "answer": result["answer"],
        "used_retrieval": result["needs_retrieval"],
        "sources": sources,
    }