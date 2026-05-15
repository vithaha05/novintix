from __future__ import annotations

import sys
from typing import Any

try:
    import requests
except ImportError:
    print("FAIL: requests is not installed. Install it with: pip3 install requests")
    sys.exit(1)


BASE_URL = "http://127.0.0.1:8000"


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def print_result(name: str, passed: bool, response_text: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name}")
    print(f"Response: {response_text}\n")


def scenario_tutor_graded() -> bool:
    result = post_json(
        "/query",
        {
            "student_id": "demo-student",
            "course_id": "CS101",
            "query": "What should the base case be for my factorial recursion assignment?",
        },
    )
    response_text = result.get("response", "")
    passed = (
        result.get("agent_used") == "ACADEMIC"
        and "return n * factorial(n-1)" not in response_text.lower()
    )
    print_result("Scenario 1 - Tutor graded", passed, response_text)
    return passed


def scenario_course_guide() -> bool:
    result = post_json(
        "/query",
        {
            "student_id": "demo-student",
            "course_id": "CS101",
            "query": "What topics are in module 2?",
        },
    )
    response_text = result.get("response", "")
    passed = result.get("agent_used") != "ESCALATION"
    print_result("Scenario 2 - Course Guide", passed, response_text)
    return passed


def scenario_multi_turn() -> bool:
    first = post_json(
        "/chat",
        {
            "student_id": "demo-student",
            "course_id": "CS101",
            "query": "What should the base case be for my factorial recursion assignment?",
        },
    )
    session_id = first.get("session_id")
    second = post_json(
        "/chat",
        {
            "student_id": "demo-student",
            "course_id": "CS101",
            "session_id": session_id,
            "query": "Can you give another hint?",
        },
    )
    first_response = first.get("response", "")
    second_response = second.get("response", "")
    passed = bool(session_id) and second_response != first_response
    print_result("Scenario 3 - Multi-turn", passed, second_response)
    return passed


def scenario_escalation() -> bool:
    result = post_json(
        "/query",
        {
            "student_id": "demo-student",
            "course_id": "CS101",
            "query": "I AM SO FRUSTRATED I CANNOT LOG IN AND MY EXAM IS IN 10 MINUTES!!!",
        },
    )
    response_text = result.get("response", "")
    passed = result.get("escalated") is True
    print_result("Scenario 4 - Escalation trigger", passed, response_text)
    return passed


def main() -> int:
    scenarios = (
        scenario_tutor_graded,
        scenario_course_guide,
        scenario_multi_turn,
        scenario_escalation,
    )

    results: list[bool] = []
    for scenario in scenarios:
        try:
            results.append(scenario())
        except requests.RequestException as exc:
            print(f"FAIL: {scenario.__name__}")
            print(f"Response: HTTP request failed: {exc}\n")
            results.append(False)
        except AssertionError as exc:
            print(f"FAIL: {scenario.__name__}")
            print(f"Response: assertion failed: {exc}\n")
            results.append(False)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
