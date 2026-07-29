"""Conversational Q&A engine with per-user context."""

from typing import Dict, List, Set

from config import COMPANIES, MIN_RELEVANCE_SCORE, TOP_K
from rag.vector_store import VectorStore


class ConversationSession:
    def __init__(self, user_email: str, allowed_companies: Set[str]):
        self.user_email = user_email
        self.allowed_companies = allowed_companies
        self.history: List[Dict[str, str]] = []

    def add_turn(self, question: str, answer: str, sources: List[Dict[str, str]]) -> None:
        self.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )

    def clear(self) -> None:
        self.history.clear()


class QAEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, user_email: str, allowed_companies: Set[str]) -> ConversationSession:
        if user_email not in self.sessions:
            self.sessions[user_email] = ConversationSession(user_email, allowed_companies)
        return self.sessions[user_email]

    def _contextualize_query(self, session: ConversationSession, question: str) -> str:
        if not session.history:
            return question

        recent = session.history[-2:]
        context_lines = []
        for turn in recent:
            context_lines.append(f"Previous question: {turn['question']}")
            context_lines.append(f"Previous answer: {turn['answer']}")

        context = "\n".join(context_lines)
        return f"{context}\nFollow-up question: {question}"

    def _build_answer(self, question: str, results: List[Dict[str, str]]) -> str:
        if not results:
            return (
                "No relevant information was found in your authorized documents. "
                "You may not have access to documents covering this topic."
            )

        best = results[0]
        company_name = COMPANIES.get(best["company_id"], best["company_id"])
        excerpt = best["text"]

        if "revenue" in question.lower():
            return (
                f"Based on {company_name}'s earnings documents, here is the relevant excerpt:\n\n"
                f"{excerpt}"
            )

        return (
            f"From {company_name}'s earnings call ({best['filename']}):\n\n"
            f"{excerpt}"
        )

    def ask(self, user_email: str, allowed_companies: Set[str], question: str) -> Dict[str, object]:
        session = self.get_or_create_session(user_email, allowed_companies)
        contextual_query = self._contextualize_query(session, question)
        results = self.vector_store.search(
            contextual_query,
            allowed_companies=allowed_companies,
            top_k=TOP_K,
        )
        results = [item for item in results if item["score"] >= MIN_RELEVANCE_SCORE]
        answer = self._build_answer(question, results)
        session.add_turn(question, answer, results)

        return {
            "answer": answer,
            "sources": results,
            "contextual_query": contextual_query,
        }

    def clear_session(self, user_email: str) -> None:
        if user_email in self.sessions:
            self.sessions[user_email].clear()
