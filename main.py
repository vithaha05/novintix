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
from fastapi.responses import FileResponse
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
    if collection.count() == 0:
        documents = _syllabus_documents()
        collection.add(
            ids=[item[0] for item in documents],
            documents=[item[1] for item in documents],
            metadatas=[item[2] for item in documents],
            embeddings=[[0.0] * 16 for _ in documents],
        )

    app.state.chroma_client = client
    app.state.course_collection = collection
    app.state.graph = edu_agent_graph
    app.state.llm = _initialize_llm()
    app.state.log_dir = log_dir


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "vector_db": "chromadb"}


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
    return response


@app.post("/chat", response_model=QueryResponse)
def chat(request: ChatRequest) -> QueryResponse:
    session_id = request.session_id or str(uuid4())
    conversation_history = session_store.get(session_id, [])
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
    session_store[session_id].append({"role": "assistant", "content": assistant_message})

    return QueryResponse(
        agent_used=result.get("intent", "UNKNOWN"),
        response=assistant_message or "No response generated",
        escalated=bool(result.get("escalated", False)),
        session_id=session_id,
    )




@app.get("/chat/{session_id}/history")
def chat_history(session_id: str) -> list:
    return session_store.get(session_id, [])
