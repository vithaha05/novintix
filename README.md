# EduAgent — Multi-Agent EdTech Support System

## Overview
EduAgent is a multi-agent AI orchestration platform for 100,000+ students. Built on FastAPI + LangGraph with Llama 3.3 70B (Groq free tier) and ChromaDB for local RAG. Routes student queries to specialized agents that enforce pedagogical boundaries, resolve technical issues, and escalate distress signals automatically.

## Live Demo
GitHub: https://github.com/vithaha05/novintix

## Agent Architecture
- Orchestrator: Zero-shot intent classification (ACADEMIC / TECH / ADMIN / UNKNOWN). Frustration detection via all-caps ratio + exclamation count + keyword scoring. Auto-escalates if frustration > 0.55 or confidence < 85%.
- Tutor Agent: Socratic mode enforced. TF-IDF cosine similarity detects graded assignments (threshold > 0.35) and activates restricted mode. Hint counter — after 3 hints on same topic, student redirected to lecture video.
- Course Guide: RAG on CS101_syllabus.json via ChromaDB. Top-3 chunk retrieval as LLM context. Eliminates hallucinations on dates and deadlines.
- Tech Support: Resolves platform issues. Auto-escalates after 3 failed troubleshooting steps.
- Escalation Agent: Packages full conversation context for human agent with priority tag.

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph |
| LLM | Llama 3.3 70B via Groq (free tier) |
| Vector DB | ChromaDB (fully local) |
| Assignment Matching | scikit-learn TF-IDF |
| API | FastAPI + Uvicorn |
| Session Memory | In-memory session_store (multi-turn /chat) |
| Testing | pytest + unittest.mock (9/9 passing) |

## Guardrails
- Academic Integrity: TF-IDF similarity > 0.35 on graded assignments forces Socratic mode
- Direct Answer Blocking: Regex classifier blocks solutions/code in tutor responses
- PII Masking: Names, emails, IDs masked before every LLM call (PIIMiddleware)
- Prompt Injection: Input filtering in core/guardrails.py

## API Endpoints
- POST /query — stateless single-turn query
- POST /chat — stateful multi-turn with session memory
- GET /chat/{session_id}/history — full conversation history
- GET /health — system health check

## Setup
1. Clone: git clone https://github.com/vithaha05/novintix.git && cd novintix/edu-agent
2. Install: pip install -r requirements.txt
3. Configure: cp .env.example .env — add GROQ_API_KEY from console.groq.com
4. Run: uvicorn main:app --reload

## Demo
python3 demo.py  # runs 4 automated scenarios, prints PASS/FAIL

## Test
pytest tests/ -v  # 9/9 tests, no Groq key required

## Folder Structure
edu-agent/
├── agents/
│   ├── orchestrator/     # StateGraph, intent classifier, frustration detector
│   ├── tutor/            # Socratic mode, TF-IDF graded detection
│   ├── course_guide/     # ChromaDB RAG
│   └── tech_support/     # Troubleshooting + escalation logic
├── core/
│   ├── guardrails.py     # Direct-answer classifier, prompt injection filter
│   ├── privacy.py        # PII masking middleware
│   └── routing.py        # Thresholds and intent categories
├── data/
│   └── syllabus/         # CS101_syllabus.json (RAG source)
├── tests/                # pytest coverage (guardrails, privacy, routing)
├── frontend/             # index.html chat UI
├── demo.py               # Automated 4-scenario demo script
└── main.py               # FastAPI app, /query, /chat, /health

## Demo Results
- Scenario 1 (Tutor graded): PASS — Socratic response, no answer leaked
- Scenario 2 (Course Guide RAG): PASS — Module 2 topics retrieved from syllabus
- Scenario 3 (Multi-turn): PASS — Session ID preserved across turns
- Scenario 4 (Escalation): PASS — Frustration detected, human handoff triggered
