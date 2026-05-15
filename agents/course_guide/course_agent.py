from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from agents.orchestrator.state import EduAgentState


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
SYLLABUS_PATH = BASE_DIR / "data" / "syllabus" / "CS101_syllabus.json"
LOGGER = logging.getLogger(__name__)


def _load_cs101_syllabus() -> dict[str, Any]:
    try:
        with SYLLABUS_PATH.open("r", encoding="utf-8") as syllabus_file:
            return json.load(syllabus_file)
    except FileNotFoundError:
        LOGGER.warning("CS101 syllabus file not found at %s", SYLLABUS_PATH)
    except json.JSONDecodeError as exc:
        LOGGER.warning("CS101 syllabus file is invalid JSON: %s", exc)
    return {}


CS101_SYLLABUS = _load_cs101_syllabus()


SYSTEM_PROMPT = """You are EduAgent's Course Guide agent.
Answer student questions using only the retrieved course context and syllabus facts.
Be concise, accurate, and student-friendly.
If the retrieved context does not contain enough information, say what is missing and suggest checking with course staff.
Do not invent deadlines, grades, policies, or assignment requirements."""


def _query_course_collection(collection: Any, query: str, n_results: int = 3) -> list[str]:
    if collection is None:
        return []

    try:
        result = collection.query(query_texts=[query], n_results=n_results)
    except Exception as exc:
        LOGGER.warning("ChromaDB course_docs query failed: %s", exc)
        return []

    documents = result.get("documents", [])
    if not documents:
        return []

    first_result = documents[0]
    if not isinstance(first_result, list):
        return [str(document) for document in documents if document]
    return [str(document) for document in first_result if document]


def _syllabus_summary() -> str:
    if not CS101_SYLLABUS:
        return "Syllabus metadata unavailable."

    course_id = CS101_SYLLABUS.get("course_id", "CS101")
    course_name = CS101_SYLLABUS.get("course_name", "Unknown course")
    modules = CS101_SYLLABUS.get("modules", [])
    module_titles = [str(module.get("title")) for module in modules if module.get("title")]
    return f"{course_id} - {course_name}. Modules: {', '.join(module_titles)}."


def get_course_response(state: EduAgentState) -> EduAgentState:
    query = state["masked_query"]
    collection = state.get("chroma_collection")
    llm = state.get("llm")

    retrieved_chunks = _query_course_collection(collection, query, n_results=3)
    context = "\n\n".join(f"Context {index + 1}: {chunk}" for index, chunk in enumerate(retrieved_chunks))
    if not context:
        context = "No relevant ChromaDB context was retrieved."

    prompt = (
        f"Syllabus summary: {_syllabus_summary()}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Student query: {query}\n\n"
        "Write the best course-guide response for the student."
    )

    if llm is None:
        response = (
            "I found the course context, but the LLM is not initialized. "
            f"Relevant course information: {retrieved_chunks[0] if retrieved_chunks else 'No matching course document found.'}"
        )
        return {**state, "agent_response": response}

    try:
        result = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        response = str(result.content)
    except Exception as exc:
        LOGGER.warning("Course Guide LLM call failed: %s", exc)
        response = (
            "I could not generate a course-guide response right now. "
            f"Relevant course information: {retrieved_chunks[0] if retrieved_chunks else 'No matching course document found.'}"
        )

    return {**state, "agent_response": response}
