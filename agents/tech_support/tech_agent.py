from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.orchestrator.state import EduAgentState


SYSTEM_PROMPT = (
    "You are EduAgent's technical support assistant. Provide concise, practical troubleshooting "
    "steps for login, upload, browser, and student portal issues."
)


def run_tech_agent(state: EduAgentState) -> EduAgentState:
    llm = state.get("llm")
    if llm is None:
        response = (
            "Try refreshing the portal, checking your browser permissions, and resetting your password "
            "if login is failing. If the issue continues, include the exact error message for support."
        )
        return {**state, "agent_used": "tech_support", "response": response}

    result = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["query"])])
    return {**state, "agent_used": "tech_support", "response": result.content}
