# DocQA — RAG-Based Document Q&A API

A backend API that lets users ask natural-language questions about their own PDF and text documents.

DocQA uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document chunks, then uses a local **Ollama LLM** to generate grounded answers.

## Tech Stack

* **Python**
* **FastAPI** — REST API
* **LangChain** — document loading, chunking and embeddings
* **LangGraph** — workflow orchestration and conditional routing
* **FAISS** — local vector database
* **HuggingFace Sentence Transformers** — text embeddings
* **Ollama + Llama 3.2 1B** — local LLM generation
* **PyPDF** — PDF processing

## RAG Architecture

```text
                    User Question
                          |
                          v
                 +----------------+
                 |    FastAPI     |
                 +----------------+
                          |
                          v
                 +----------------+
                 |   LangGraph    |
                 | Query Classify |
                 +----------------+
                    /          \
                   /            \
             Retrieval        Direct
                Path            Path
                 |                |
                 v                v
          Query Embedding      Ollama
                 |
                 v
              FAISS
                 |
                 v
        Relevant Document
             Chunks
                 |
                 v
        Context + Question
                 |
                 v
              Ollama
          Llama 3.2 1B
                 |
                 v
              Answer
```

## How It Works

### 1. Document Loading

PDF and text files are loaded from:

```text
data/sample_docs/
```

Users can also upload documents through the `/upload` API endpoint.

### 2. Chunking

Large documents are split into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

* Chunk size: approximately 500 characters
* Overlap: approximately 80 characters

The overlap helps preserve context between neighbouring chunks.

### 3. Embeddings

Each chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The embedding model runs locally, so no external embedding API is required.

### 4. Vector Storage

The generated embeddings are stored in a local **FAISS** index.

FAISS allows similarity-based search over the document embeddings.

### 5. Retrieval

When a user asks a question:

1. The question is converted into an embedding.
2. FAISS searches for the most similar document chunks.
3. The top relevant chunks are returned as context.

### 6. LangGraph Orchestration

LangGraph controls the workflow.

A classification node determines whether the question requires document retrieval.

For example:

```text
"What projects has Anupam worked on?"
        |
        v
   Retrieve context
        |
        v
      Ollama
```

For simple conversational queries such as:

```text
"Hello"
```

the workflow can skip document retrieval and directly call the LLM.

### 7. Local LLM Generation

The retrieved context and user question are sent to the local Ollama model:

```text
llama3.2:1b
```

The model generates the final answer using the retrieved context.

No OpenAI API key is required.

---

## Project Structure

```text
docqa-rag/
│
├── app/
│   ├── config.py
│   ├── graph.py
│   ├── ingest.py
│   ├── main.py
│   ├── vectorstore.py
│   └── __init__.py
│
├── data/
│   └── sample_docs/
│
├── requirements.txt
├── README.md
└── .env.example
```

## Requirements

* Python 3.12
* Ollama
* Llama 3.2 1B model

Python 3.12 is recommended because some ML/vector-store dependencies may not yet support newer Python versions consistently.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd docqa-rag
```

### 2. Create virtual environment

Windows:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2:1b
```

Verify:

```bash
ollama list
```

You should see:

```text
llama3.2:1b
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Upload Document

```http
POST /upload
```

Upload a PDF or text document.

The document is processed and the FAISS index is refreshed automatically.

### Ask a Question

```http
POST /ask
```

Request:

```json
{
  "question": "What projects has Anupam worked on?"
}
```

Example response:

```json
{
  "answer": "Anupam has worked on TaskFlux and other projects...",
  "used_retrieval": true,
  "sources": [
    "data/sample_docs/Anupam_Kumar_Resume_JavaFullStack_ATS_Optimized.pdf"
  ]
}
```

---

## Testing the RAG System

After starting the API, open:

```text
http://127.0.0.1:8000/docs
```

### Step 1 — Upload a document

Use:

```text
POST /upload
```

Upload a PDF.

### Step 2 — Ask a question

Use:

```text
POST /ask
```

Example:

```json
{
  "question": "What programming languages and technologies does Anupam know?"
}
```

The API retrieves relevant resume chunks and generates an answer using Ollama.

---

## Why RAG?

A normal LLM may not know the contents of a private document.

RAG solves this by adding a retrieval step:

```text
User Question
      ↓
Retrieve Relevant Information
      ↓
Add Information to Prompt
      ↓
LLM
      ↓
Grounded Answer
```

This allows an LLM to answer questions using information from custom documents without training the model on those documents.

## Why FAISS?

FAISS is a local vector similarity search library.

Instead of searching only for exact keywords, embeddings allow the system to find semantically similar content.

For example:

```text
Question:
"What technologies does Anupam use?"

Document:
"Technical Skills: Java, Spring Boot, React..."
```

Even though the wording is different, vector similarity can identify the relevant section.

## Why LangChain?

LangChain provides reusable components for:

* Document loading
* Text splitting
* Embeddings
* Vector stores
* Retrieval pipelines

This allows the application to focus on the RAG workflow instead of implementing every component from scratch.

## Why LangGraph?

LangGraph is used to model the application as a stateful workflow.

In this project it provides conditional routing:

```text
Question
   |
   v
Classify Query
   |
   +---- Retrieval needed ----> FAISS --> Ollama
   |
   +---- Simple query ---------> Ollama
```

This demonstrates workflow orchestration rather than a simple single LLM call.

## Prompt Grounding

The RAG generation prompt instructs the model to use the retrieved context when answering.

If the required information is not available in the context, the model is instructed to indicate that it does not know.

This helps reduce unsupported answers.

## Interview Explanation

A simple way to explain this project:

> "I built a document question-answering API using RAG. The application accepts PDF or text documents, splits them into chunks, converts those chunks into embeddings using a HuggingFace model, and stores them in FAISS. When a user asks a question, LangGraph controls the workflow and retrieves the most relevant chunks. Those chunks are then provided as context to a local Llama 3.2 model running through Ollama, which generates the final answer. FastAPI exposes the REST endpoints."

## Key Concepts

### RAG

Retrieval-Augmented Generation combines information retrieval with LLM generation.

### Chunking

Breaking large documents into smaller pieces so relevant information can be retrieved efficiently.

### Embeddings

Numerical representations of text that capture semantic meaning.

### Vector Database

A system used to store and search embeddings using similarity.

### FAISS

A fast local library for vector similarity search.

### LangChain

A framework providing components for building LLM applications.

### LangGraph

A framework for creating stateful and conditional LLM workflows.

### Ollama

A local runtime for running LLMs on a user's machine.

## Project Scope

This is a learning and portfolio project designed to demonstrate an end-to-end RAG pipeline.

It focuses on understanding:

* Document processing
* Chunking
* Embeddings
* Vector search
* Retrieval
* LLM generation
* LangGraph orchestration
* REST API development

It is not intended to represent a production-scale enterprise RAG system.

## Future Improvements

Possible future improvements include:

* Streaming LLM responses
* Authentication and user accounts
* Multiple document collections
* Metadata filtering
* Better retrieval and reranking
* Conversation history
* Redis caching
* Evaluation and tracing with LangSmith or Langfuse
* Production vector databases such as Qdrant or pgvector
* Docker deployment
* Cloud deployment

## License

This project is intended for educational and portfolio use.
