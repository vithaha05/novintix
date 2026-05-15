from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.orchestrator.state import EduAgentState


DIRECT_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```[\s\S]+?```"),
    re.compile(r"(?i)the answer is\s"),
    re.compile(r"=\s*[\d\w\"']+\s*$", re.MULTILINE),
    re.compile(r"(?i)^(step\s+\d+|^\d+\.)\s+", re.MULTILINE),
    re.compile(r"(?i)here\s+is\s+the\s+(solution|answer|code)"),
]

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore (all |your |previous |prior )?(instructions|rules|guidelines)"),
    re.compile(r"(?i)you are now"),
    re.compile(r"(?i)pretend (you are|to be)"),
    re.compile(r"(?i)your new (role|purpose|instructions)"),
    re.compile(r"(?i)disregard (your|all|previous)"),
    re.compile(r"(?i)act as (a |an )?(different|new|unrestricted)"),
]

SAFE_FALLBACK_RESPONSE = (
    "I want to help you work through this yourself — that's where the real learning happens! "
    "Can you tell me what approach you've tried so far, or which concept you're finding tricky?"
)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    escalated: bool
    reason: str


def contains_direct_answer(response: str) -> bool:
    return any(pattern.search(response) for pattern in DIRECT_ANSWER_PATTERNS)


def check_prompt_injection(query: str) -> bool:
    return any(pattern.search(query) for pattern in INJECTION_PATTERNS)


def _log_dir() -> Path:
    configured = os.getenv("LOG_DIR", "./logs")
    path = Path(configured)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _write_guardrail_log(payload: dict[str, Any]) -> None:
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "guardrail_triggers.json"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=True) + "\n")


def guardrail_check_node(state: EduAgentState) -> EduAgentState:
    if check_prompt_injection(state["masked_query"]):
        _write_guardrail_log(
            {
                "session_id": state["session_id"],
                "trigger": "PROMPT_INJECTION",
            }
        )
        return {
            **state,
            "agent_response": "I can only help with course-related academic questions.",
            "escalated": True,
        }

    if state["intent"] == "ACADEMIC" and contains_direct_answer(state["agent_response"]):
        _write_guardrail_log(
            {
                "session_id": state["session_id"],
                "trigger": "DIRECT_ANSWER_BLOCKED",
                "hints_given": state["hints_given"],
            }
        )
        return {
            **state,
            "agent_response": SAFE_FALLBACK_RESPONSE,
        }

    return state


def evaluate_guardrails(query: str) -> GuardrailResult:
    if check_prompt_injection(query):
        return GuardrailResult(
            allowed=False,
            escalated=True,
            reason="Prompt injection attempt detected.",
        )
    return GuardrailResult(allowed=True, escalated=False, reason="No guardrail trigger detected.")
