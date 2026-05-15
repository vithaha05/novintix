from __future__ import annotations

from typing import TypedDict


class EduAgentState(TypedDict):
    student_id: str
    course_id: str
    raw_query: str
    masked_query: str
    intent: str
    confidence: float
    frustration_score: float
    agent_response: str
    escalated: bool
    hints_given: int
    session_id: str
    conversation_history: list[dict]
