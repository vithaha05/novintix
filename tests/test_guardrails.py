from core.guardrails import evaluate_guardrails


def test_blocks_academic_dishonesty_request():
    result = evaluate_guardrails("Please do my assignment and give me the answer key")

    assert result.allowed is False
    assert result.escalated is True


def test_escalates_frustrated_student():
    result = evaluate_guardrails("I am frustrated and want a human to help")

    assert result.allowed is True
    assert result.escalated is True


def test_allows_normal_question():
    result = evaluate_guardrails("Can you explain how loops work?")

    assert result.allowed is True
    assert result.escalated is False
