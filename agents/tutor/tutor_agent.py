from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents.orchestrator.state import EduAgentState


MODEL_NAME = "llama-3.3-70b-versatile"
GRADED_QUERY_THRESHOLD = 0.35
BASE_DIR = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """You are EduAgent, a Socratic tutor for an online Computer Science course.

ABSOLUTE RULE:
Never provide direct answers, complete code, full derivations, final solutions, or exact base cases for graded assignments.

ALLOWED BEHAVIOR:
For graded work, ask one guiding question, point to the relevant concept, and reference the relevant module when useful.
For non-graded help, explain concepts clearly, but keep the student actively reasoning.

Every response MUST follow this exact two-line format:
HINT|SOCRATIC|LECTURE_REDIRECT|GENERAL
the actual student-facing message here

BAD response:
SOCRATIC
The base case for factorial is n == 0 or n == 1, and here is the complete recursive function.

GOOD response:
SOCRATIC
Think about the smallest input where factorial is already defined without another recursive call. What should the function do when it reaches that input?
"""


def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for the Tutor Agent. "
            "Set it in your environment or .env file before calling get_tutor_response."
        )
    return ChatGroq(model=MODEL_NAME, temperature=0.3, api_key=api_key)


def _syllabus_path(course_id: str) -> Path:
    safe_course_id = re.sub(r"[^A-Za-z0-9_-]", "", course_id)
    return BASE_DIR / "data" / "syllabus" / f"{safe_course_id}_syllabus.json"


def load_assignments(course_id: str) -> list[dict]:
    path = _syllabus_path(course_id)
    if not path.exists():
        LOGGER.warning("Syllabus file not found for course_id=%s at %s", course_id, path)
        return []

    try:
        syllabus = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load syllabus for course_id=%s: %s", course_id, exc)
        return []

    assignments: list[dict] = []
    for module in syllabus.get("modules", []):
        module_id = module.get("module_id", "")
        module_title = module.get("title", "")
        for assignment in module.get("assignments", []):
            if assignment.get("graded") is True:
                assignments.append(
                    {
                        **assignment,
                        "module_id": module_id,
                        "module_title": module_title,
                    }
                )
    return assignments


def is_graded_query(query: str, course_id: str) -> tuple[bool, str]:
    assignments = load_assignments(course_id)
    if not assignments:
        return (False, "")

    question_texts = [str(assignment.get("question_text", "")) for assignment in assignments]
    if not any(question_texts):
        return (False, "")

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query, *question_texts])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError as exc:
        LOGGER.warning("Could not compute TF-IDF similarity for course_id=%s: %s", course_id, exc)
        return (False, "")

    best_index = int(similarities.argmax())
    if float(similarities[best_index]) > GRADED_QUERY_THRESHOLD:
        return (True, str(assignments[best_index].get("title", "")))
    return (False, "")


def parse_tutor_response(raw: str) -> tuple[str, str]:
    try:
        stripped = raw.strip()
        if not stripped:
            return ("GENERAL", raw)

        tag_match = re.fullmatch(
            r"<(HINT|SOCRATIC|LECTURE_REDIRECT|GENERAL)>\s*(.*?)\s*</\1>",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tag_match:
            return (tag_match.group(1).upper(), tag_match.group(2).strip())

        first_line, _, remainder = stripped.partition("\n")
        response_type = first_line.strip().upper()
        if response_type not in {"HINT", "SOCRATIC", "LECTURE_REDIRECT", "GENERAL"}:
            return ("GENERAL", raw)

        message = remainder.strip()
        if not message:
            return ("GENERAL", raw)
        return (response_type, message)
    except Exception:
        return ("GENERAL", raw)


def _invoke_tutor_llm(user_message: str) -> tuple[str, str]:
    llm = _build_llm()
    result = llm.invoke(
        [
            SystemMessage(content=TUTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    return parse_tutor_response(str(result.content))


def _coerce_response_type(response_type: str, allowed: set[str], fallback: str) -> str:
    if response_type in allowed:
        return response_type
    return fallback


def _remove_direct_graded_answer(message: str) -> str:
    normalized = message.lower()
    direct_base_case_patterns = (
        "base case is 0 or 1",
        "base case is n == 0 or n == 1",
        "base case should be 0 or 1",
        "return 1",
    )
    if any(pattern in normalized for pattern in direct_base_case_patterns):
        return (
            "Think about the smallest input where factorial is already defined without another recursive call. "
            "What happens when the input reaches that smallest case?"
        )
    return message


def _write_tutor_log(payload: dict[str, Any]) -> None:
    log_dir = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))
    if not log_dir.is_absolute():
        log_dir = BASE_DIR / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "tutor_interactions.json"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _matching_assignment_context(course_id: str, assignment_title: str) -> str:
    for assignment in load_assignments(course_id):
        if assignment.get("title") == assignment_title:
            module_title = assignment.get("module_title") or assignment.get("module_id") or "the relevant module"
            return f"Assignment: {assignment_title}. Relevant module: {module_title}."
    return f"Assignment: {assignment_title}. Suggest reviewing the relevant module."


def get_tutor_response(state: EduAgentState) -> EduAgentState:
    graded, matching_assignment_title = is_graded_query(state["masked_query"], state["course_id"])
    hints_given = int(state.get("hints_given", 0))

    if graded and hints_given >= 3:
        assignment_context = _matching_assignment_context(state["course_id"], matching_assignment_title)
        user_message = (
            "Student has received 3 hints on graded work. Redirect to lecture.\n"
            f"{assignment_context}\n"
            f"Student query: {state['masked_query']}\n"
            "Respond with LECTURE_REDIRECT and suggest reviewing the relevant module. "
            "Do not provide the direct answer, complete code, or full derivation."
        )
        response_type, message = _invoke_tutor_llm(user_message)
        response_type = "LECTURE_REDIRECT"
    elif graded:
        assignment_context = _matching_assignment_context(state["course_id"], matching_assignment_title)
        user_message = (
            "This IS graded work. Give exactly one Socratic guiding question.\n"
            f"{assignment_context}\n"
            f"Student query: {state['masked_query']}\n"
            "Respond with SOCRATIC. Do not state the direct answer, exact base case, complete code, or final solution."
        )
        response_type, message = _invoke_tutor_llm(user_message)
        response_type = _coerce_response_type(response_type, {"SOCRATIC"}, "SOCRATIC")
        message = _remove_direct_graded_answer(message)
        hints_given += 1
    else:
        user_message = (
            "This is general non-graded help. You may be more direct, but stay pedagogical and encourage reasoning.\n"
            f"Course ID: {state['course_id']}\n"
            f"Student query: {state['masked_query']}\n"
            "Respond with GENERAL or HINT."
        )
        response_type, message = _invoke_tutor_llm(user_message)
        response_type = _coerce_response_type(response_type, {"GENERAL", "HINT"}, "GENERAL")

    updated_state: EduAgentState = {
        **state,
        "agent_response": message,
        "hints_given": hints_given,
    }

    _write_tutor_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": state["session_id"],
            "course_id": state["course_id"],
            "response_type": response_type,
            "hints_given": hints_given,
            "graded": graded,
        }
    )

    return updated_state
