from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from core.routing import ESCALATION_THRESHOLD, FRUSTRATION_THRESHOLD, IntentCategory, RouteDecision


CLASSIFIER_SYSTEM_PROMPT = """You are EduAgent's intent classifier.
Classify the student query into exactly one of these intents:
ACADEMIC, TECH, ADMIN, UNKNOWN.

Use ACADEMIC for learning help, concepts, tutoring, homework guidance, and course topic questions.
Use TECH for login, password, upload, browser, access, account, and platform issues.
Use ADMIN for deadlines, grading, rubrics, syllabus, submissions, schedules, and policy questions.
Use UNKNOWN when the query does not clearly fit any category.

You must respond in exactly this format:
INTENT: <ACADEMIC | TECH | ADMIN | UNKNOWN>
CONFIDENCE: <0.0 to 1.0>
REASONING: <brief reason>
"""

FRUSTRATION_KEYWORDS = (
    "useless",
    "broken",
    "give up",
    "cant do this",
    "not working",
    "failed",
    "hopeless",
)


def _parse_intent(raw_intent: str) -> IntentCategory:
    normalized = raw_intent.strip().upper()
    if normalized == "ACADEMIC":
        return IntentCategory.ACADEMIC
    if normalized == "TECH":
        return IntentCategory.TECH
    if normalized == "ADMIN":
        return IntentCategory.ADMIN
    return IntentCategory.UNKNOWN


def _clamp_confidence(value: str) -> float:
    try:
        confidence = float(value.strip())
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, confidence))


def classify_intent(query: str, llm: ChatGroq) -> RouteDecision:
    response = llm.invoke(
        [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )
    content = str(response.content)
    parsed: dict[str, str] = {}

    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().upper()] = value.strip()

    category = _parse_intent(parsed.get("INTENT", "UNKNOWN"))
    confidence = _clamp_confidence(parsed.get("CONFIDENCE", "0.0"))
    reasoning = parsed.get("REASONING", "No reasoning provided by classifier.")

    return RouteDecision(category=category, confidence=confidence, reasoning=reasoning)


def detect_frustration(query: str) -> float:
    score = 0.0
    words = re.findall(r"\b[A-Za-z]+\b", query)
    if words:
        all_caps_words = [word for word in words if word.isupper() and len(word) > 1]
        if len(all_caps_words) / len(words) > 0.3:
            score += 0.3

    if query.count("!") > 2:
        score += 0.2

    normalized = query.lower()
    keyword_hits = sum(keyword in normalized for keyword in FRUSTRATION_KEYWORDS)
    score += min(0.4, keyword_hits * 0.2)

    return min(1.0, score)


def should_escalate(route: RouteDecision, frustration: float) -> bool:
    return route.confidence < ESCALATION_THRESHOLD or frustration > FRUSTRATION_THRESHOLD
