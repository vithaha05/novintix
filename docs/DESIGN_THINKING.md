# EduAgent — Design Thinking Decode
**Double Diamond Framework | May 2026**

## 1. Discover (Empathize)
**Objective**: Understand friction in the current 100k+ student support ecosystem.

**Student Friction**
- "I'm stuck at 2 AM and the forum response takes 2 days."
- "I'm worried I'll fail because I can't log in to the exam portal."
- "I just need a hint but the AI keeps giving me the full answer."

**Staff Friction**
- "I've answered the same grading rubric question 50 times today."
- "I can't tell who needs a hint vs who just wants the answer."
- "High-value escalations drown in password reset tickets."

**Platform Friction**
- Rising support costs — avg 14-hour MTTR
- High student churn from delayed responses
- No differentiation between tier-1 (repeat) and tier-2 (complex) queries

## 2. Define (Synthesize)

**Personalization HMW**: How Might We provide instant, tailored academic support that mimics a 1-on-1 tutor without doing the work for the student?

**Efficiency HMW**: How Might We automate 90% of technical and administrative queries without losing the human touch for complex issues?

**Integrity HMW**: How Might We ensure AI support aids genuine learning rather than providing a shortcut to completion?

## 3. Develop (Ideate)

**Multi-Agent Orchestration**: Specialized agents instead of one generic bot. Orchestrator classifies intent; specialists handle domain with strict per-agent guardrails.

**Socratic Tutoring Mode**: Prompt-engineering layer forcing guiding questions. TF-IDF graded assignment detection activates restricted mode automatically if query matches active graded work.

**Sentiment-Driven Escalation**: Frustration scorer using all-caps ratio + exclamation density + keyword hits. Score > 0.55 bypasses agents and routes to human. Tunable without code changes.

**RAG-based Course Guide**: ChromaDB local vector store ingests CS101_syllabus.json. Top-3 chunk retrieval as LLM context. Eliminates hallucinations on dates.

**PII Boundary Layer**: All queries masked before LLM. Pseudonymous tokens replace real names, emails, IDs. Separate audit log.

**Architecture Decision Table**:
| Decision | Chosen | Rejected | Reason |
|----------|--------|----------|--------|
| LLM | Groq Llama 3.3 70B | OpenAI GPT-4o | Free tier |
| Vector DB | ChromaDB (local) | Pinecone | No API key, data on device |
| Assignment Match | TF-IDF cosine | sentence-transformers | No GPU, ARM Mac |
| Orchestration | LangGraph StateGraph | LangChain AgentExecutor | Explicit state, easier debug |
| Session Memory | In-memory dict | Redis | Zero infra for demo |

## 4. Deliver (Implement)

**Phase 1 — Foundation**: FastAPI + LangGraph + ChromaDB scaffold. Tech Support and Course Guide agents. /health endpoint and basic routing proven.

**Phase 2 — Core Intelligence**: Orchestrator with zero-shot classification. Tutor Agent with Socratic mode and TF-IDF graded detection. Factorial demo working end-to-end.

**Phase 3 — Guardrails & Privacy**: PII masking middleware. Direct-answer classifier. Prompt injection detection. All guardrail trigger logs wired.

**Phase 4 — Production Hardening**: Multi-turn /chat with session memory. 9/9 pytest passing. 4/4 demo scenarios automated. Frontend chat UI.

**Proven Outcomes**:
- FastAPI: /query, /chat, /health, /chat/{id}/history
- LangGraph: 4-node StateGraph (classify → tutor/course/tech/escalate)
- Tutor: Socratic, TF-IDF graded detection, hint counter, PII-safe logging
- Course Guide: ChromaDB RAG, top-3 chunk retrieval
- Frustration: all-caps + exclamation + keywords, threshold 0.55
- PII: masks names, emails, IDs before every LLM call
- Tests: 9/9 passing, all mocked, no Groq key required
- Demo: 4/4 passing
- GitHub: github.com/vithaha05/novintix

**Evaluation**:
- Weekly human-in-the-loop audit of sampled responses
- Escalation rate target < 5%, tunable via FRUSTRATION_THRESHOLD
- Re-query rate proxy: same question within 24h = previous response failed
