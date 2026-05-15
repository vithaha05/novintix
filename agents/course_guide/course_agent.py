from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents.orchestrator.state import EduAgentState


def _collection_documents(collection) -> list[str]:
    if collection is None:
        return []

    result = collection.get(include=["documents"])
    return result.get("documents", []) or []


def _best_context(query: str, documents: list[str]) -> str:
    if not documents:
        return ""

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([query, *documents])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    best_index = int(scores.argmax())
    return documents[best_index]


def run_course_agent(state: EduAgentState) -> EduAgentState:
    documents = _collection_documents(state.get("chroma_collection"))
    context = _best_context(state["query"], documents)
    if context:
        response = f"Here is the most relevant course information I found: {context}"
    else:
        response = "I could not find matching course information yet. Please check the syllabus or contact course staff."

    return {**state, "agent_used": "course_guide", "response": response}
