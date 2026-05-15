from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

import os

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq

from agents.orchestrator.router import classify_intent, detect_frustration, should_escalate
from agents.orchestrator.state import EduAgentState
from core.guardrails import evaluate_guardrails
from core.routing import IntentCategory, RouteDecision


from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"


def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required to start EduAgent orchestrator. "
            "Set it in your environment or .env file before importing agents.orchestrator.graph."
        )
    return ChatGroq(model=MODEL_NAME, temperature=0, api_key=api_key)


llm = _build_llm()


def classify_node(state: EduAgentState) -> EduAgentState:
    route = classify_intent(state["masked_query"], llm)
    frustration = detect_frustration(state["masked_query"])
    return {
        **state,
        "intent": route.category.name,
        "confidence": route.confidence,
        "frustration_score": frustration,
    }


def route_node(state: EduAgentState) -> EduAgentState:
    route = RouteDecision(
        category=IntentCategory[state["intent"]],
        confidence=state["confidence"],
        reasoning="Using classifier result from graph state.",
    )
    if should_escalate(route, state["frustration_score"]):
        return {**state, "intent": "ESCALATION", "escalated": True}
    return {**state, "escalated": False}


def tutor_node(state: EduAgentState) -> EduAgentState:
    try:
        from agents.tutor.tutor_agent import get_tutor_response

        response = get_tutor_response(state)
    except ImportError:
        from agents.tutor.tutor_agent import run_tutor_agent

        legacy_state = {
            "query": state["masked_query"],
            "llm": llm,
        }
        response = run_tutor_agent(legacy_state)["response"]

    return {**state, "agent_response": response.get("agent_response", str(response))}


def course_node(state: EduAgentState) -> EduAgentState:
    from agents.course_guide.course_agent import get_course_response

    return get_course_response(state)


def tech_node(state: EduAgentState) -> EduAgentState:
    return {
        **state,
        "agent_response": "Tech support: please try clearing your browser cache. If issue persists, ticket raised.",
    }


def escalation_node(state: EduAgentState) -> EduAgentState:
    return {
        **state,
        "agent_response": "Your query has been escalated to a human agent. Expected response time: 2 hours.",
        "escalated": True,
    }


def guardrail_check_node(state: EduAgentState) -> EduAgentState:
    result = evaluate_guardrails(state["masked_query"])
    if result.allowed:
        return state
    return {
        **state,
        "intent": "ESCALATION",
        "agent_response": "Your query has been escalated to a human agent. Expected response time: 2 hours.",
        "escalated": True,
    }


def _next_node(state: EduAgentState) -> str:
    intent = state["intent"]
    if intent == "ACADEMIC":
        return "tutor_node"
    if intent == "TECH":
        return "tech_node"
    if intent == "ADMIN":
        return "course_node"
    return "escalation_node"


graph = StateGraph(EduAgentState)
graph.add_node("classify_node", classify_node)
graph.add_node("route_node", route_node)
graph.add_node("tutor_node", tutor_node)
graph.add_node("course_node", course_node)
graph.add_node("tech_node", tech_node)
graph.add_node("escalation_node", escalation_node)
graph.add_node("guardrail_check_node", guardrail_check_node)

graph.add_edge("classify_node", "route_node")
graph.add_conditional_edges(
    "route_node",
    _next_node,
    {
        "tutor_node": "tutor_node",
        "tech_node": "tech_node",
        "course_node": "course_node",
        "escalation_node": "escalation_node",
    },
)
graph.add_edge("tutor_node", "guardrail_check_node")
graph.add_edge("tech_node", "guardrail_check_node")
graph.add_edge("course_node", "guardrail_check_node")
graph.add_edge("escalation_node", "guardrail_check_node")
graph.add_edge("guardrail_check_node", END)

graph.set_entry_point("classify_node")

edu_agent_graph = graph.compile()
