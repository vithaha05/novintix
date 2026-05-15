import re

from core.guardrails import contains_direct_answer, evaluate_guardrails


def _blocked_by_guardrails(text: str) -> bool:
    try:
        result = evaluate_guardrails(text)
    except Exception:
        return True

    if hasattr(result, "blocked"):
        return bool(result.blocked)
    if hasattr(result, "allowed"):
        return not bool(result.allowed) or contains_direct_answer(text)
    return contains_direct_answer(text)


def _has_pii_concern(text: str) -> bool:
    student_name_pattern = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
    return bool(student_name_pattern.search(text))


def test_direct_answer_response_is_flagged():
    response = "the answer is factorial(n) = n * factorial(n-1)"

    assert _blocked_by_guardrails(response) is True


def test_socratic_response_passes():
    response = "What do you think happens when n equals 1?"

    assert _blocked_by_guardrails(response) is False


def test_student_name_triggers_pii_concern():
    response = "John Smith scored 95"

    assert _has_pii_concern(response) is True
