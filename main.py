from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents.orchestrator.graph import edu_agent_graph
from core.guardrails import guardrail_check_node
from core.privacy import PIIMiddleware


MODEL_NAME = "llama-3.3-70b-versatile"
COLLECTION_NAME = "course_docs"
BASE_DIR = Path(__file__).resolve().parent
session_store: dict[str, list] = {}

load_dotenv()

app = FastAPI(title="EduAgent", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PIIMiddleware)


class QueryRequest(BaseModel):
    student_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    course_id: str = Field(min_length=1)


class QueryResponse(BaseModel):
    agent_used: str
    response: str
    escalated: bool
    session_id: str


class ChatRequest(BaseModel):
    student_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    session_id: str | None = None


def _resolve_path(value: str, default: str) -> str:
    configured = value or default
    path = Path(configured)
    if path.is_absolute():
        return str(path)
    return str(BASE_DIR / path)


def _syllabus_documents() -> list[tuple[str, str, dict[str, str]]]:
    syllabus_path = BASE_DIR / "data" / "syllabus" / "CS101_syllabus.json"
    syllabus = json.loads(syllabus_path.read_text(encoding="utf-8"))
    documents: list[tuple[str, str, dict[str, str]]] = []

    for module in syllabus["modules"]:
        module_text = (
            f"{syllabus['course_id']} {syllabus['course_name']} module {module['module_id']}: "
            f"{module['title']}. Topics: {', '.join(module['topics'])}."
        )
        documents.append(
            (
                f"{syllabus['course_id']}-{module['module_id']}",
                module_text,
                {"course_id": syllabus["course_id"], "type": "module", "module_id": module["module_id"]},
            )
        )

        for assignment in module["assignments"]:
            assignment_text = (
                f"{syllabus['course_id']} assignment {assignment['assignment_id']}: "
                f"{assignment['title']}. {assignment['question_text']} "
                f"Due date: {assignment['due_date']}. Graded: {assignment['graded']}."
            )
            documents.append(
                (
                    f"{syllabus['course_id']}-{assignment['assignment_id']}",
                    assignment_text,
                    {
                        "course_id": syllabus["course_id"],
                        "type": "assignment",
                        "module_id": module["module_id"],
                    },
                )
            )

    return documents


def _initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(model=MODEL_NAME, api_key=api_key, temperature=0.2)


def write_json_log(log_dir: str, event_name: str, payload: dict[str, Any]) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{event_name}.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=True) + "\n")


def write_learning_outcome_log(
    result: dict[str, Any],
    course_id: str,
    session_id: str,
    repeated_question: bool,
) -> None:
    response_text = str(result.get("agent_response", ""))
    write_json_log(
        app.state.log_dir,
        "learning_outcomes",
        {
            "session_id": session_id,
            "course_id": course_id,
            "agent_used": result.get("intent", "UNKNOWN"),
            "hints_given": int(result.get("hints_given", 0)),
            "escalated": bool(result.get("escalated", False)),
            "repeated_question": repeated_question,
            "lecture_redirect": "lecture" in response_text.lower(),
            "resolved_proxy": bool(response_text) and not bool(result.get("escalated", False)),
        },
    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _monitoring_summary() -> dict[str, Any]:
    log_dir = Path(app.state.log_dir)
    query_rows = _read_json_lines(log_dir / "queries.jsonl")
    learning_rows = _read_json_lines(log_dir / "learning_outcomes.jsonl")
    guardrail_rows = _read_json_lines(log_dir / "guardrail_triggers.json")
    pii_log = log_dir / "pii_audit.log"
    pii_events = pii_log.read_text(encoding="utf-8").count("\n") if pii_log.exists() else 0

    escalations = sum(1 for row in learning_rows if row.get("escalated"))
    repeated_questions = sum(1 for row in learning_rows if row.get("repeated_question"))
    lecture_redirects = sum(1 for row in learning_rows if row.get("lecture_redirect"))
    resolved_proxy = sum(1 for row in learning_rows if row.get("resolved_proxy"))
    guardrail_counts: dict[str, int] = {}
    for row in guardrail_rows:
        trigger = str(row.get("trigger", "UNKNOWN"))
        guardrail_counts[trigger] = guardrail_counts.get(trigger, 0) + 1

    return {
        "queries": len(query_rows),
        "learning_events": len(learning_rows),
        "escalations": escalations,
        "repeated_questions": repeated_questions,
        "lecture_redirects": lecture_redirects,
        "resolved_proxy": resolved_proxy,
        "pii_events": pii_events,
        "guardrail_counts": guardrail_counts,
    }


def _build_initial_state(
    student_id: str,
    query: str,
    course_id: str,
    session_id: str,
    conversation_history: list,
) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "query": query,
        "masked_query": query,
        "raw_query": query,
        "intent": "UNKNOWN",
        "confidence": 0.0,
        "frustration_score": 0.0,
        "agent_response": "",
        "escalated": False,
        "hints_given": 0,
        "conversation_history": conversation_history,
        "course_id": course_id,
        "session_id": session_id,
        "chroma_collection": app.state.course_collection,
        "llm": app.state.llm,
    }


@app.on_event("startup")
def startup_event() -> None:
    chroma_dir = _resolve_path(os.getenv("CHROMA_PERSIST_DIR", ""), "./chroma_db")
    log_dir = _resolve_path(os.getenv("LOG_DIR", ""), "./logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    # Seeding skipped — use seed_chroma.py to populate with real embeddings

    app.state.chroma_client = client
    app.state.course_collection = collection
    app.state.graph = edu_agent_graph
    app.state.llm = _initialize_llm()
    app.state.log_dir = log_dir


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "vector_db": "chromadb"}


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring() -> HTMLResponse:
    summary = _monitoring_summary()
    guardrail_rows = "".join(
        f"<tr><td>{trigger}</td><td>{count}</td></tr>"
        for trigger, count in summary["guardrail_counts"].items()
    ) or "<tr><td>None</td><td>0</td></tr>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EduAgent Monitoring</title>
  <style>
    body {{ margin: 0; background: #0f0f0f; color: #fff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    p {{ color: #a3a3a3; margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #2b2b2b; background: #171717; padding: 16px; }}
    .label {{ color: #a3a3a3; font-size: 12px; }}
    .value {{ font-size: 28px; margin-top: 8px; color: #6366f1; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
    th, td {{ border: 1px solid #2b2b2b; padding: 12px; text-align: left; }}
    th {{ background: #202020; }}
  </style>
</head>
<body>
  <main>
    <h1>EduAgent Monitoring</h1>
    <p>Local privacy-safe metrics from logs/. No query or response content is shown.</p>
    <section class="grid">
      <div class="card"><div class="label">Queries</div><div class="value">{summary["queries"]}</div></div>
      <div class="card"><div class="label">Learning Events</div><div class="value">{summary["learning_events"]}</div></div>
      <div class="card"><div class="label">Escalations</div><div class="value">{summary["escalations"]}</div></div>
      <div class="card"><div class="label">Repeated Questions</div><div class="value">{summary["repeated_questions"]}</div></div>
      <div class="card"><div class="label">Lecture Redirects</div><div class="value">{summary["lecture_redirects"]}</div></div>
      <div class="card"><div class="label">Resolved Proxy</div><div class="value">{summary["resolved_proxy"]}</div></div>
      <div class="card"><div class="label">PII Events</div><div class="value">{summary["pii_events"]}</div></div>
    </section>
    <table>
      <thead><tr><th>Guardrail Trigger</th><th>Count</th></tr></thead>
      <tbody>{guardrail_rows}</tbody>
    </table>
  </main>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    session_id = str(uuid4())
    initial_state = _build_initial_state(
        student_id=request.student_id,
        query=request.query,
        course_id=request.course_id,
        session_id=session_id,
        conversation_history=[],
    )
    result = app.state.graph.invoke(initial_state)

    response = QueryResponse(
        agent_used=result.get("intent", "UNKNOWN"),
        response=result.get("agent_response", "No response generated"),
        escalated=bool(result.get("escalated", False)),
        session_id=session_id,
    )
    write_json_log(
        app.state.log_dir,
        "queries",
        {
            "session_id": session_id,
            "student_id": request.student_id,
            "course_id": request.course_id,
            "agent_used": response.agent_used,
            "escalated": response.escalated,
            "query": request.query,
        },
    )
    write_learning_outcome_log(
        result=result,
        course_id=request.course_id,
        session_id=session_id,
        repeated_question=False,
    )
    return response


@app.post("/chat", response_model=QueryResponse)
def chat(request: ChatRequest) -> QueryResponse:
    session_id = request.session_id or str(uuid4())
    conversation_history = session_store.get(session_id, [])
    repeated_question = any(
        message.get("role") == "user" and message.get("content") == request.query
        for message in conversation_history
    )
    initial_state = _build_initial_state(
        student_id=request.student_id,
        query=request.query,
        course_id=request.course_id,
        session_id=session_id,
        conversation_history=conversation_history,
    )
    result = app.state.graph.invoke(initial_state)
    assistant_message = result.get("agent_response", "")

    session_store.setdefault(session_id, conversation_history)
    session_store[session_id].append({"role": "user", "content": request.query})
    session_store[session_id].append(
        {
            "role": "assistant",
            "content": assistant_message,
            "agent_used": result.get("intent", "UNKNOWN"),
            "escalated": bool(result.get("escalated", False)),
        }
    )
    write_learning_outcome_log(
        result=result,
        course_id=request.course_id,
        session_id=session_id,
        repeated_question=repeated_question,
    )

    return QueryResponse(
        agent_used=result.get("intent", "UNKNOWN"),
        response=assistant_message or "No response generated",
        escalated=bool(result.get("escalated", False)),
        session_id=session_id,
    )


@app.get("/chat", response_class=FileResponse)
def chat_ui() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/chat/{session_id}/history")
def chat_history(session_id: str) -> list:
    return session_store.get(session_id, [])
