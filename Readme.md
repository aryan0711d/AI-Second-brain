# 🧠 AI Second Brain

A full-stack RAG (Retrieval-Augmented Generation) application that acts as your personal AI-powered knowledge assistant. Save notes, ask questions, and get intelligent answers grounded in your own data — with real-time streaming responses.

**Live Demo:** [ai-second-brain-frontend.vercel.app](https://ai-second-brain-frontend.vercel.app)

---

## What It Does

Most AI chatbots answer from general training data. AI Second Brain answers from **your** notes. You store knowledge, and the AI retrieves and reasons over it. This is the core idea behind RAG — making AI responses accurate, personal, and grounded in real context.

---

## Architecture

```
User → React Frontend (Vercel)
          ↓
     FastAPI Backend (Render)
          ↓
  ┌───────────────────────┐
  │  Cohere Embed API     │  ← converts text to 384-dim vectors
  └───────────────────────┘
          ↓
  ┌───────────────────────┐
  │  MongoDB Atlas        │  ← vector search + full-text search (hybrid)
  └───────────────────────┘
          ↓
  ┌───────────────────────┐
  │  Groq LLM (streaming) │  ← generates answer from retrieved context
  └───────────────────────┘
          ↓
     Streamed response back to user
```

---

## Key Features

- **Hybrid Search** — combines semantic vector search and keyword-based full-text search, then re-ranks results by combined score for higher retrieval accuracy
- **Streaming Responses** — answers stream token-by-token in real time using Server-Sent Events
- **Session Management** — multiple named chat sessions, each with independent conversation history
- **JWT Authentication** — secure user registration and login with bcrypt password hashing and JWT tokens
- **Per-user Data Isolation** — all notes and history are scoped to the authenticated user
- **Persistent Chat History** — conversation context is stored in MongoDB and injected into each LLM call

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Render |
| Embeddings | Cohere `embed-english-light-v3.0` (384-dim) |
| Vector DB | MongoDB Atlas with `$vectorSearch` |
| LLM | Groq API (llama/mixtral) with streaming |
| Auth | JWT via `python-jose`, bcrypt via `passlib` |

---

## How Hybrid Search Works

When you ask a question, the backend runs two searches in parallel:

1. **Vector Search** — your question is embedded into a 384-dimensional vector and compared against all your stored note embeddings using cosine similarity via MongoDB Atlas `$vectorSearch`
2. **Full-text Search** — MongoDB's text index finds notes with direct keyword matches

Results from both are merged and re-ranked by combined score. The top 3 notes are injected into the LLM prompt as context. This hybrid approach outperforms either method alone — vector search catches semantic matches, text search catches exact terms.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/register` | Create new user account |
| POST | `/login` | Authenticate and receive JWT token |
| POST | `/add-note` | Save a note with embedding |
| POST | `/ask` | Ask a question (streaming response) |
| DELETE | `/history/{session_id}` | Clear a specific session's history |
| DELETE | `/history` | Clear all chat history for the user |

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB Atlas account (free tier works)
- Groq API key (free)
- Cohere API key (free trial)

### Backend Setup

```bash
git clone https://github.com/yourusername/ai-second-brain
cd ai-second-brain/backend

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
MODEL_NAME=llama3-8b-8192
MONGO_URL=your_mongodb_atlas_url
SECRET_KEY=your_random_secret_key
COHERE_API_KEY=your_cohere_key
```

```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### MongoDB Atlas Vector Index

Create a vector search index named `vector_index` on the `notes` collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

---

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI app, all routes and logic
│   ├── requirements.txt
│   └── .env              # not committed
├── frontend/
│   ├── src/
│   │   └── App.jsx       # entire React frontend
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Design Decisions

**Why hybrid search instead of just vector search?**
Vector search alone misses exact keyword matches — if you save a note with a specific name or number, semantic search might not surface it. Text search alone misses paraphrased or conceptually similar content. Combining both gives better recall.

**Why Groq for the LLM?**
Groq's inference hardware is significantly faster than standard API providers — important for streaming UX where latency is visible to the user.

**Why Cohere for embeddings?**
Reliable, consistent API with a free tier. The `embed-english-light-v3.0` model produces 384-dimensional vectors — same dimensionality as the widely-used `all-MiniLM-L6-v2` — with no cold-start issues unlike serverless inference endpoints.

**Why stream the response?**
RAG answers can take 3-10 seconds to generate. Streaming makes the experience feel instant — users see tokens appearing rather than staring at a loading spinner.

---

## Limitations & Known Improvements

- Vector search pre-filter by `user_id` should be applied inside `$vectorSearch` using the `filter` parameter rather than as a post-match stage — current implementation searches all vectors then filters, which is less efficient at scale
- No rate limiting on API endpoints
- No pagination on chat history retrieval
- Frontend sessions are stored in localStorage only — clearing browser data loses session names

---

## Demo Video

- https://youtu.be/DupGU-jVZ_g

---

## License

MIT