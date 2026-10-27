# Palm Mind AI – RAG Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion**: upload a `.pdf` or `.txt` file. The text is split into chunks (`fixed` or `recursive`), converted to embeddings and stored in **Qdrant**. Document metadata is saved in **SQLite**.
2. **Conversational RAG**: ask questions about the uploaded documents. Chat history is kept in **Redis**, so follow-up questions work. You can also **book an interview** in the chat; the LLM extracts the name, email, date and time, and the booking is saved in SQLite.

The RAG pipeline is custom built (no `RetrievalQAChain`, no FAISS or Chroma, no UI).

## Why Ollama instead of OpenAI?

The project first used the OpenAI API, but the API key returned `429 – no credits remaining`, so it couldn't create embeddings or answers. It was switched to **Ollama**, which runs the models locally for free through an OpenAI-compatible API. Only the `.env` settings changed; the code is the same. To go back to OpenAI, see the end of this file.

## Tech stack

FastAPI · Ollama (`llama3.2` for chat, `nomic-embed-text` for embeddings) · Qdrant · Redis · SQLite · PyMuPDF

## How to run

```powershell
# 1. Install packages
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com/download), then download the models
ollama pull nomic-embed-text
ollama pull llama3.2

# 3. Create the settings file (already set up for Ollama)
copy .env.example .env

# 4. Start Qdrant and Redis
docker compose up -d

# 5. Start the API
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** to test. The first request can take 10–30 seconds while the model loads.

## API usage

### Upload a document: `POST /api/v1/documents/ingest`
Form fields: `file` (`.pdf` or `.txt`) and `chunking_strategy` (`fixed` or `recursive`).

- **fixed**: equal-size pieces of text with a small overlap.
- **recursive**: splits by paragraph, then line, then sentence, so chunks keep their meaning.

```json
{ "document_id": "7e8ef7fe-...", "filename": "sample.txt", "chunking_strategy": "recursive", "chunk_count": 1, "message": "Document ingested successfully" }
```

### Ask a question: `POST /api/v1/chat`
```json
{ "session_id": "test1", "message": "How long is the interview?" }
```
```json
{ "answer": "...the AI engineering interview lasts 30 minutes.", "booking_created": false, "retrieved_chunks": 1 }
```
Use the same `session_id` to continue the conversation.

### Book an interview: `POST /api/v1/chat`
```json
{ "session_id": "test1", "message": "I want to book an interview. My name is Sachin Bista, email sachin@example.com, on 2026-11-05 at 10:30" }
```
```json
{ "booking_created": true, "booking": { "id": 1, "name": "Sachin Bista", "email": "sachin@example.com", "interview_date": "2026-11-05", "interview_time": "10:30:00", "status": "pending" } }
```
If any details are missing, the assistant asks for them and you can reply in the next message.

## How it works

- **Ingestion**: read file → extract text → chunk → embed → store in Qdrant → save metadata in SQLite.
- **Chat**: load history from Redis → embed the question → find matching chunks in Qdrant → the LLM answers from those chunks → save the messages to Redis.
- **Booking**: if the user asks to book, the LLM returns the details as JSON → validated (email, date, time) → saved to the `interview_bookings` table.

## Project structure

```
app/
├── main.py            # app startup
├── config.py          # settings from .env
├── database.py        # SQLite connection
├── models/            # database tables and request/response schemas
├── routers/           # ingestion.py and chat.py endpoints
└── services/          # chunking, embeddings/LLM, Qdrant, Redis, booking logic
```

## Switch back to OpenAI

Set these in `.env` (with a key that has credits), remove `OPENAI_BASE_URL`, then restart:
```dotenv
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
QDRANT_COLLECTION=palm_mind_documents
```

## Common problems

| Problem | Fix |
|---|---|
| `Cannot reach the model server` | Start the Ollama app. |
| `Model not found` | Run `ollama pull <model name>`. |
| `retrieved_chunks: 0` | Upload a document first. |
| `.env` changes not working | Restart uvicorn. |
