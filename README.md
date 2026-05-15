# EduAgent — Multi-Agent EdTech Support System

EduAgent is a FastAPI + LangGraph multi-agent support system for an online education platform. It routes student queries to specialized agents for academic tutoring, course guidance, technical support, or escalation, while enforcing academic-integrity and privacy guardrails.

GitHub: https://github.com/vithaha05/novintix

## What Is Implemented

- FastAPI backend with `/query`, `/chat`, `/chat/{session_id}/history`, and `/health`
- LangGraph `StateGraph` orchestrator
- Groq Llama 3.3 70B integration through `langchain-groq`
- Local ChromaDB course-document retrieval
- TF-IDF graded-assignment matching with scikit-learn
- Socratic Tutor Agent with hint counting
- Course Guide Agent with ChromaDB RAG over `CS101_syllabus.json`
- Tech Support response path
- Escalation path for low confidence or high frustration
- PII masking middleware
- Prompt-injection and direct-answer guardrails
- In-memory multi-turn chat sessions
- Vanilla HTML/CSS/JS frontend at `/chat`
- Pytest coverage for routing, privacy, and guardrails
- Automated demo script with four scenarios

## Project Structure

```text
edu-agent/
├── agents/
│   ├── orchestrator/     # LangGraph state, router, graph
│   ├── tutor/            # Socratic tutor + TF-IDF graded detection
│   ├── course_guide/     # ChromaDB RAG course guide
│   └── tech_support/     # Technical support response logic
├── core/
│   ├── guardrails.py     # Direct-answer and prompt-injection checks
│   ├── privacy.py        # PII masking middleware
│   └── routing.py        # Intent categories and thresholds
├── data/
│   ├── syllabus/         # CS101_syllabus.json
│   └── faqs/             # General FAQ seed data
├── docs/
│   ├── PRD.md
│   └── DESIGN_THINKING.md
├── frontend/
│   └── index.html        # Browser chat UI
├── tests/
├── demo.py
├── main.py
├── requirements.txt
└── .env.example
```

## Prerequisites

- Python 3.10+ recommended
- A free Groq API key from https://console.groq.com
- No paid APIs required
- No Docker required
- No GPU required

## Setup

From this repository:

```bash
cd /Users/apple/Desktop/life/university/placement/projects/novintix/edu-agent
```

Or after cloning:

```bash
git clone https://github.com/vithaha05/novintix.git
cd novintix/edu-agent
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and add:

```bash
GROQ_API_KEY=your_groq_key_here
CHROMA_PERSIST_DIR=./chroma_db
LOG_DIR=./logs
APP_ENV=development
```

Important: the app needs `GROQ_API_KEY` because the LangGraph orchestrator initializes Groq on startup.

## Run The Backend

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok","model":"llama-3.3-70b-versatile","vector_db":"chromadb"}
```

## Open The Chat UI

After the server is running, open:

```text
http://127.0.0.1:8000/chat
```

If you see `{"detail":"Method Not Allowed"}`, restart the server with `uvicorn main:app --reload`. That means the old server process has not picked up the `GET /chat` frontend route yet.

## API Usage

Stateless single-turn query:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S001",
    "course_id": "CS101",
    "query": "What topics are in module 2?"
  }'
```

Stateful chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S001",
    "course_id": "CS101",
    "query": "What should the base case be for my factorial recursion assignment?"
  }'
```

The response includes a `session_id`. Pass it back for the next turn:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S001",
    "course_id": "CS101",
    "session_id": "PASTE_SESSION_ID_HERE",
    "query": "Can you give another hint?"
  }'
```

View chat history:

```bash
curl http://127.0.0.1:8000/chat/PASTE_SESSION_ID_HERE/history
```

## Run The Demo

Start the server first, then in a second terminal:

```bash
python3 demo.py
```

The demo runs four scenarios:

- Tutor graded-assignment query
- Course Guide RAG query
- Multi-turn `/chat` session
- Frustration-based escalation

It prints `PASS` or `FAIL` for each scenario and exits with code `0` only if all pass.

## Run Tests

The tests mock LLM calls and do not require a Groq key:

```bash
pytest tests/ -v
```

If `pytest` is missing:

```bash
pip install -r requirements.txt
```

## Logs

Local logs are written under `logs/`:

- `logs/queries.jsonl` — query metadata
- `logs/tutor_interactions.json` — tutor metadata only, no query/response content
- `logs/pii_audit.log` — PII detection metadata only
- `logs/guardrail_triggers.json` — prompt-injection/direct-answer trigger metadata

No LangSmith or external monitoring service is used.

## Problem Statement Completion

Source document: `Problem_Statement_3 (2).pdf`

| Requirement From PDF | Status | Evidence |
|---|---:|---|
| Understand student/platform challenges | Complete | `docs/DESIGN_THINKING.md` discover/define sections |
| Define an HMW problem statement | Complete | `docs/PRD.md` and `docs/DESIGN_THINKING.md` |
| Orchestrator manages agents | Complete | `agents/orchestrator/graph.py` uses LangGraph `StateGraph` |
| Course guidance agent | Complete | `agents/course_guide/course_agent.py` uses ChromaDB RAG |
| Assignment help agent | Complete | `agents/tutor/tutor_agent.py` uses Socratic mode and TF-IDF graded matching |
| Technical support agent | Mostly complete | TECH route returns troubleshooting guidance; retry-count escalation after 3 failed tech steps is not fully state-tracked |
| Escalation agent/path | Complete | Graph routes low-confidence/frustrated queries to escalation response |
| Prevent academic dishonesty | Complete | Tutor prompt, TF-IDF graded detection, direct-answer guardrail |
| Avoid direct answers to graded assignments | Complete | Socratic mode and `contains_direct_answer` guardrail |
| Ensure student data privacy | Mostly complete | Email/phone/ID/Aadhaar masking exists; name masking is documented/test-assisted but basic regex name masking is not fully implemented in `core/privacy.py` |
| Monitoring layer tracks misuse/escalation | Mostly complete | Local JSON/log files track PII and guardrail triggers; no dashboard |
| Track learning outcomes | Partial | Hint count and interaction metadata exist; no real learning-outcome analytics |
| PRD deliverable | Complete | `docs/PRD.md` |
| GitHub repository with README | Complete | This repository and README |
| Design-thinking decode | Complete | `docs/DESIGN_THINKING.md` |
| Chat history link | Complete locally | `GET /chat/{session_id}/history` |

## Honest Status

The main candidate-task implementation is complete enough to run and demo end to end: backend, orchestration, agents, guardrails, privacy layer, frontend, tests, and demo script are present.

The only parts I would label as not fully production-grade are:

- Tech Support does not yet track “3 failed troubleshooting steps” across turns.
- Monitoring is local log files, not a dashboard or analytics service.
- Learning-outcome tracking is represented by metadata and hint counts, not actual course-performance analytics.
- Name PII masking is not as strong as email/phone/student-ID masking.

For the assessment PDF, the build satisfies the requested prototype and documentation deliverables; the remaining gaps are production-hardening items rather than missing core scaffold.
