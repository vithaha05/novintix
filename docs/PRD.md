# EduAgent — Product Requirements Document
**Version 1.0 | May 2026**

## 1. Overview
EduAgent is a multi-agent AI orchestration platform for 100,000+ students. Built on FastAPI + LangGraph + Llama 3.3 70B (Groq free tier) + ChromaDB local RAG.

## 2. Problem Statement
Platform suffers 14-hour avg MTTR. Students bypass learning by seeking direct answers externally.

**How Might We** provide immediate, high-quality, pedagogically sound support at scale without increasing headcount or compromising academic integrity?

## 3. Agent Architecture

### 3.1 Orchestrator
- Logic: Zero-shot LLM intent classification (ACADEMIC / TECH / ADMIN / UNKNOWN)
- Frustration Detection: all-caps ratio + exclamation count + keyword hits → score > 0.55 = escalate
- Escalation Trigger: confidence < 85% OR frustration > 0.55

### 3.2 Course Guidance Agent
- Scope: Syllabus, schedules, prerequisites, learning objectives
- Implementation: ChromaDB RAG on CS101_syllabus.json, top-3 chunk retrieval
- Output: Hallucination-resistant answers grounded in course data

### 3.3 Assignment Help Agent (Tutor)
- Policy: STRICT NO-DIRECT-ANSWER — Socratic mode enforced
- Detection: TF-IDF cosine similarity against graded assignment corpus (threshold > 0.35)
- Hint Counter: After 3 hints on same topic → redirect to lecture video

### 3.4 Technical Support Agent
- Scope: Logins, video playback, submission errors
- Escalation: After 3 failed troubleshooting steps

### 3.5 Escalation Agent
- Role: Human-in-the-loop handoff
- Output: Full conversation context + priority tag for human agent

### 3.6 Sample Interaction
Student: "What should the base case be for my factorial recursion assignment?"
Orchestrator: ACADEMIC intent → TF-IDF graded match → Tutor (restricted mode)
Tutor: "At what point can the factorial be determined without another recursive call?"
Student: "Oh, return 1 when n is 0 or 1?"
Tutor: "Exactly! Now how would you express that as a stopping condition in code?"

## 4. Guardrails

### 4.1 Academic Integrity
- TF-IDF similarity > 0.35 on active graded item → Socratic mode forced
- Regex classifier blocks direct answers, code solutions, step-by-step derivations

### 4.2 Scaffolding
- No code snippets or direct formulas for graded work
- hints_given tracked in EduAgentState — after 3 → redirect to lecture

### 4.3 Privacy & Security
- PII Masking: names, emails, IDs masked before every LLM call via PIIMiddleware
- Audit Logging: logs/pii_audit.log and logs/guardrail_triggers.json
- Retention: 30 days, then anonymized
- Prompt Injection: input filtering in core/guardrails.py

## 5. Tech Stack
| Layer | Technology | Reason |
|-------|-----------|--------|
| Orchestration | LangGraph StateGraph | Explicit state, full routing control |
| LLM | Llama 3.3 70B via Groq | Free tier, fast, OpenAI-compatible |
| Vector DB | ChromaDB (local) | No API key, data on device |
| Assignment Match | scikit-learn TF-IDF | No GPU, ARM Mac compatible |
| API | FastAPI + Uvicorn | Async, auto OpenAPI docs |
| Session Memory | In-memory dict | Stateful multi-turn /chat |
| Testing | pytest + unittest.mock | Deterministic, no network calls |

## 6. Success Metrics
| Metric | Target | Baseline |
|--------|--------|---------|
| Mean Time to Resolution | < 1 minute | 14 hours |
| Escalation Rate | < 5% | ~35% (manual) |
| System Accuracy | > 98% | Not tracked |
| Demo Scenarios | 4/4 pass | 4/4 passing |
| Test Coverage | 9/9 pass | 9/9 passing |

## 7. Risks & Mitigations
- Hallucinations on dates → ChromaDB RAG grounds answers in course JSON
- Prompt Injection → hardened system prompts + guardrails.py filtering
- Over-escalation → frustration threshold 0.55, tunable in core/routing.py
- Integrity bypass → TF-IDF detection + Socratic classifier layer
