from types import SimpleNamespace
from unittest.mock import Mock

from agents.orchestrator.router import classify_intent
from core.routing import IntentCategory


def _mock_llm(intent: str, confidence: float = 0.91) -> Mock:
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content=f"INTENT: {intent}\nCONFIDENCE: {confidence}\nREASONING: mocked classifier response"
    )
    return llm


def test_login_submission_query_routes_to_tech_or_escalation():
    decision = classify_intent("I can't log in to submit my assignment", _mock_llm("TECH"))

    assert decision.category in {IntentCategory.TECH, IntentCategory.ESCALATION}
    assert decision.confidence == 0.91


def test_recursion_base_case_query_routes_to_academic():
    decision = classify_intent("What is the base case for recursion?", _mock_llm("ACADEMIC"))

    assert decision.category == IntentCategory.ACADEMIC
    assert decision.confidence == 0.91


def test_assignment_due_date_query_routes_to_admin_or_course():
    decision = classify_intent("When is the M2 assignment due?", _mock_llm("ADMIN"))

    assert decision.category in {IntentCategory.ADMIN}
    assert decision.confidence == 0.91
