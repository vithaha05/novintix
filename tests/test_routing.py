from agents.orchestrator.router import classify_intent
from core.routing import IntentCategory


def test_routes_academic_query():
    decision = classify_intent("Can you explain recursion and algorithm complexity?")

    assert decision.category == IntentCategory.ACADEMIC
    assert decision.confidence > 0.5


def test_routes_tech_query():
    decision = classify_intent("I cannot access the portal login page")

    assert decision.category == IntentCategory.TECH
    assert decision.confidence > 0.5


def test_routes_admin_query():
    decision = classify_intent("What is the deadline for the gradebook assignment?")

    assert decision.category == IntentCategory.ADMIN
    assert decision.confidence > 0.5
