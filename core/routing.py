from dataclasses import dataclass
from enum import Enum


class IntentCategory(str, Enum):
    ACADEMIC = "academic"
    TECH = "tech"
    ADMIN = "admin"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RouteDecision:
    category: IntentCategory
    confidence: float
    reasoning: str


ESCALATION_THRESHOLD = 0.85
FRUSTRATION_THRESHOLD = 0.70
