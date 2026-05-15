# EduAgent

EduAgent is a local-first educational assistant scaffold built with FastAPI, LangGraph, Groq, ChromaDB, and scikit-learn TF-IDF retrieval. It avoids paid or hosted observability services and writes plain JSON logs to the local `logs/` directory.

## Stack

- Orchestration: LangGraph `StateGraph`
- LLM: Groq `llama-3.3-70b-versatile` through `langchain-groq`
- Vector DB: local persistent ChromaDB
- Backend: FastAPI and Uvicorn
- Logging: local JSONL files in `logs/`
- Similarity: scikit-learn TF-IDF

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a Groq API key to `.env`:

```bash
GROQ_API_KEY=your_key_here
```

The app can still start without a Groq key, but agent responses fall back to deterministic local messages.

## Run

```bash
uvicorn main:app --reload
```

## Endpoints

```http
GET /health
```

Returns service status, model name, and vector database name.

```http
POST /query
Content-Type: application/json

{
  "student_id": "student-123",
  "query": "When is the gradebook assignment due?",
  "course_id": "CS101"
}
```

Returns:

```json
{
  "agent_used": "course_guide",
  "response": "Here is the most relevant course information I found: ...",
  "escalated": false,
  "session_id": "..."
}
```

## Test

```bash
pytest
```
