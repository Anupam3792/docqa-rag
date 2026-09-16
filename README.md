# DocQA — RAG-based Document Q&A API

A backend service that lets you ask natural-language questions over your own
documents. Retrieves relevant chunks with a vector DB, then generates a
grounded answer — with a LangGraph node that routes trivial queries away
from retrieval to save cost/latency.

## Stack
FastAPI · LangChain · LangGraph · FAISS (Vector DB) · HuggingFace Sentence
Transformers (embeddings) · Ollama (local LLM, free, generation)

## How it works (RAG pipeline)
1. **Load** — `.txt` / `.pdf` files from `data/sample_docs/`
2. **Chunk** — `RecursiveCharacterTextSplitter` splits docs into ~500-char
   overlapping pieces (overlap = 80 chars) so context isn't cut mid-sentence
3. **Embed** — each chunk -> 384-dim vector via `all-MiniLM-L6-v2`
   (runs locally, no API cost)
4. **Store** — vectors go into a **FAISS** index, persisted to disk
5. **Retrieve** — on a query, embed the question and pull the top-k (default 4)
   most similar chunks via cosine/L2 similarity search
6. **Orchestrate (LangGraph)** — a small graph first classifies whether the
   query even needs retrieval (skips it for greetings/small talk), then
   routes to either the RAG path or a direct-answer path
7. **Generate** — retrieved chunks are stuffed into a prompt and sent to an
   LLM to produce a grounded answer

## Setup
```bash
cd docqa-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional: change OLLAMA_MODEL
```

**One-time Ollama setup (free, runs locally):**
1. Install Ollama from https://ollama.com
2. Pull a model: `ollama pull llama3.2` (or a smaller `llama3.2:1b` for a
   low-RAM machine)
3. Start the server: `ollama serve` (keep this running in a separate terminal)

## Run (local)
```bash
# 1. Build the vector index from data/sample_docs/
python -m app.ingest

# 2. Start the API
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for interactive Swagger UI, or
**http://127.0.0.1:8000/ for a simple chat UI** to talk to the API directly.

## Run with Docker (recommended for the resume line "Docker")
```bash
docker compose up --build
```
This starts two containers: `ollama` (pulls `llama3.2` automatically on
first run) and `docqa-api` (the FastAPI service, index built at image-build
time). API is available at http://localhost:8000/docs — no local Python
install needed at all.

## Try it
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leaves do employees get?", "session_id": "test-1"}'
```
Send a follow-up with the **same `session_id`** and it remembers the earlier
turn (conversation memory via LangGraph's checkpointer):
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What about during probation?", "session_id": "test-1"}'
```
Notice this second question makes no sense on its own — the model resolves
"What about" using the conversation history stored under `session_id`.

Add your own PDFs/text files via the `/upload` endpoint — it re-ingests and
refreshes the FAISS index automatically.

---

## Interview prep: how to explain each concept

**What is RAG?**
Retrieval-Augmented Generation. Instead of relying on an LLM's frozen
training data, you retrieve relevant chunks from your own documents at
query time and feed them into the prompt as context — so answers are
grounded in real, up-to-date, private data instead of the model hallucinating.

**Why chunking?**
Embedding models and LLM context windows are limited, and retrieval
precision drops on huge blocks of text. Splitting docs into overlapping
~500-char chunks keeps each piece semantically coherent while the overlap
prevents losing context that straddles a chunk boundary.

**What's a vector DB, and why FAISS?**
Traditional DBs match on exact/keyword values. A vector DB stores numeric
embeddings and finds nearest neighbours by similarity — so it can match
*meaning*, not just keywords. FAISS (from Meta) is a fast, free, local
library for this; production systems might swap it for Pinecone/Qdrant/
Weaviate/pgvector, but the underlying idea (nearest-neighbour search over
embeddings) is the same.

**What's an embedding?**
A dense numeric vector representing a piece of text's meaning. Semantically
similar text ends up close together in vector space, which is what makes
similarity search work.

**Why LangChain?**
It's the glue: standardised loaders (PDF/text), text splitters, embedding
wrappers, and vector-store integrations, so you're not hand-writing every
piece of the pipeline.

**Why LangGraph, and what's "tool orchestration"?**
LangChain chains are usually linear. LangGraph lets you define a **stateful
graph** with conditional branches — in this project, a `classify_query`
node decides whether a question needs document retrieval at all before
deciding the next step. That branching/decision-making over multiple steps
is what "tool orchestration" means in an AI-engineering context.

**What's prompt engineering here?**
The `generate_node` prompt explicitly instructs the LLM to answer *only*
from the provided context and to say "I don't know" otherwise — this
reduces hallucination and is a basic but real example of grounding a
prompt.

**LangSmith / Langfuse (mentioned on your friend's list but not wired in
here):** these are observability/tracing tools for LLM apps — they log
every prompt, retrieval, and response so you can debug and evaluate chains
in production. Be upfront in the interview that you understand *what they're
for* (tracing/observability for LLM pipelines) even if you haven't
production-deployed one yet — don't claim hands-on production use you
don't have.

**Why Ollama for the LLM step?** It runs the model entirely on your own
machine — no API key, no per-request cost, works offline. The trade-off
(worth knowing for interviews) is that local models are smaller/weaker than
hosted ones like GPT-4, and you need enough RAM to run them well.

**How does conversation memory work?**
LangGraph's `MemorySaver` checkpointer persists the full graph state keyed
by a `thread_id` (called `session_id` here). Each turn's Q&A pair is
appended to a `history` list in the state (using an `operator.add` reducer
so it accumulates instead of overwriting), and the last few turns are fed
back into the prompt on the next call — so the model can resolve follow-up
questions like "what about during probation?" using prior context. This is
different from RAG retrieval: retrieval finds relevant *document* chunks;
memory carries forward the *conversation* itself.

**Why Docker here?**
Packaging the app + its exact dependency versions into a container means it
runs identically on any machine — no "works on my laptop" issues. The
`docker-compose.yml` also runs a separate `ollama` container so the whole
stack (API + local LLM) comes up with one command, which is a realistic
example of multi-container orchestration you can speak to in an interview.

**Be honest about scope:** this is a learning project built to understand
the RAG pattern end-to-end, not a production system. That's a completely
normal and respectable thing to say in an interview — it shows you
understand what you built rather than reciting buzzwords.
