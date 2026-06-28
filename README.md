# ShopNow Customer Support Bot

An AI-powered customer support bot for e-commerce, built with a RAG (Retrieval-Augmented Generation) pipeline, FastAPI backend, and Streamlit frontend. Answers customer questions about shipping, returns, products, and orders — and automatically escalates low-confidence answers for human review.

---

## Features

- **RAG pipeline** — retrieves relevant knowledge base chunks before generating answers, grounding responses in real policy documents
- **Two-stage retrieval** — embedding search (top 10 candidates) followed by cross-encoder reranking (top 3), ensuring the most relevant chunks reach the LLM
- **Confidence scoring** — every answer is scored using the cross-encoder's relevance signal; low-confidence answers are automatically escalated
- **Escalation workflow** — flagged conversations are stored in SQLite and surfaced in the admin panel for human review
- **Admin panel** — view all conversations, resolve escalations with notes, and upload new documents to the knowledge base live
- **Document ingestion** — supports `.txt`, `.pdf`, `.docx`, and `.csv` uploads via the admin UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via [Groq](https://groq.com) (free tier) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, HuggingFace) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` (local, sentence-transformers) |
| Vector store | ChromaDB (persistent, local) |
| RAG framework | LlamaIndex 0.14 |
| Backend | FastAPI + SQLAlchemy + SQLite |
| Frontend | Streamlit |

---

## Project Structure

```
customer-support-bot/
├── backend/
│   ├── database.py      # SQLAlchemy models — Conversation, Message, Escalation
│   ├── schemas.py       # Pydantic request/response models
│   ├── rag.py           # RAG pipeline — retrieval, reranking, confidence scoring
│   └── main.py          # FastAPI app — chat, admin, and upload endpoints
├── frontend/
│   ├── Home.py          # Customer chat UI
│   └── pages/
│       └── Admin.py     # Admin panel — escalations, conversations, document upload
├── knowledge_base/
│   ├── shipping_policy.txt
│   ├── return_policy.txt
│   ├── faq.txt
│   └── products.txt
└── main.py              # Entry point
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/aabhas-gupta/customer-support-bot.git
cd customer-support-bot
uv sync
```

### 2. Create a `.env` file

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The first run indexes the knowledge base into ChromaDB. Subsequent runs load the existing index.

### 4. Run the frontend

```bash
streamlit run frontend/Home.py
```

- Customer chat: `http://localhost:8501`
- Admin panel: `http://localhost:8501/Admin`
- API docs: `http://localhost:8000/docs`

---

## How It Works

```
Customer question
      │
      ▼
Embedding search → top 10 candidates from ChromaDB
      │
      ▼
Cross-encoder reranker → top 3 most relevant chunks
      │
      ▼
LLM (Llama 3.3 70B) → generates answer from top 3 chunks
      │
      ▼
Confidence score → sigmoid over cross-encoder score
      │
      ├─ High (≥ 60%) → green badge, show sources
      ├─ Medium (35–60%) → yellow badge, show sources
      └─ Low (< 30%) → red badge, escalate to human review
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Customer chat — query the RAG pipeline |
| `GET` | `/admin/conversations` | List all conversations with messages |
| `GET` | `/admin/escalations` | List escalations (filter by resolved status) |
| `POST` | `/admin/resolve/{id}` | Mark an escalation as resolved |
| `POST` | `/admin/upload` | Upload a new document to the knowledge base |
| `GET` | `/health` | Health check |
