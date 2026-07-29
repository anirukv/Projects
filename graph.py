from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from typing import Any, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

load_dotenv()

CHROMA_DB_DIR = "/chroma_db"
COLLECTION_NAME = "earnings_calls"
GEMINI_CHAT_MODEL = "gemini-3.5-flash"
COMPANY_ALIASES = {
    "google": "Google",
    "companya": "Google",
    "netflix": "Netflix",
    "companyb": "Netflix",
    "apple": "Apple",
    "companyc": "Apple",
    "meta": "Meta",
    "companyd": "Meta",
    "amazon": "Amazon",
    "companye": "Amazon",
}


class GraphState(TypedDict):
    messages: list[dict[str, str]]
    user_email: str
    allowed_companies: list[str]
    retrieved_docs: list[dict[str, Any]]
    current_question: str
    answer: str


def _require_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Create one at https://aistudio.google.com/apikey and set it in your environment or .env file."
        )
    return api_key


def create_initial_state(user_email: str, allowed_companies: list[str], messages: list[dict[str, str]] | None = None) -> GraphState:
    return {
        "messages": messages or [],
        "user_email": user_email,
        "allowed_companies": allowed_companies,
        "retrieved_docs": [],
        "current_question": "",
        "answer": "",
    }


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


def _build_embedding(text: str, dimension: int = 512) -> list[float]:
    vector = [0.0] * dimension
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        vector[index] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "for",
    "with",
    "from",
    "about",
    "that",
    "this",
    "what",
    "which",
    "does",
    "say",
    "says",
    "their",
    "its",
    "are",
    "is",
    "be",
    "in",
    "on",
    "as",
    "it",
    "at",
    "by",
    "was",
    "were",
    "how",
    "why",
    "who",
    "when",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS and len(token) > 2]


def _score_relevance(question: str, document: str, company: str | None = None) -> float:
    question_tokens = _tokenize(question)
    doc_tokens = _tokenize(document)
    if not question_tokens or not doc_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    overlap = sum(min(3, count) for token, count in doc_counter.items() if token in set(question_tokens))
    company_bonus = 2.0 if company and company.lower() in document.lower() else 0.0
    return overlap + company_bonus


def _normalize_company_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    if cleaned in COMPANY_ALIASES:
        return COMPANY_ALIASES[cleaned]
    if cleaned.startswith("company"):
        return cleaned.replace("company", "Company", 1)
    return None


def _build_fallback_answer(question: str, retrieved_docs: list[dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "I could not find any authorized documents relevant to that question."

    best_doc = max(
        retrieved_docs,
        key=lambda doc: (doc.get("relevance", 0.0), -doc.get("distance", 0.0)),
    )
    excerpt = " ".join(best_doc.get("content", "").split())
    snippet = excerpt[:700]
    return (
        f"Based on the authorized document {best_doc.get('source', 'the source')} from {best_doc.get('company', 'the company')}, "
        f"a concise answer is: {snippet}"
    )


def _extract_text_from_response(response: Any) -> str:
    if response is None:
        return ""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]).strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def build_graph() -> StateGraph:
    """Build the explicit retrieve -> generate graph with per-user memory."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            temperature=0.2,
            google_api_key=_require_api_key(),
        )
    except Exception:
        llm = None

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        # Structural access control boundary: this metadata filter is applied at retrieval time.
        collection = _get_collection()
        if not collection or not state.get("current_question"):
            return {"retrieved_docs": []}

        question = state["current_question"]
        allowed_companies = [
            normalized_company
            for company in state.get("allowed_companies", [])
            if (normalized_company := _normalize_company_name(company)) is not None
        ]
        if not allowed_companies:
            return {"retrieved_docs": []}

        requested_company = None
        lowered_question = question.lower()
        for company_name in COMPANY_ALIASES:
            if company_name in lowered_question:
                requested_company = COMPANY_ALIASES[company_name]
                break

        if requested_company is not None and requested_company not in allowed_companies:
            return {"retrieved_docs": []}

        # Chroma's local API uses a simpler filter structure, so we query each authorized company
        # and then re-rank the returned chunks by lexical overlap with the user’s question.
        all_docs: list[dict[str, Any]] = []
        for company in allowed_companies:
            results = collection.query(
                query_embeddings=[_build_embedding(question)],
                n_results=8,
                where={"company": {"$eq": company}},
                include=["metadatas", "documents", "distances"],
            )
            for document, metadata, distance in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                content = document or ""
                meta = metadata or {}
                all_docs.append(
                    {
                        "content": content,
                        "company": meta.get("company", company),
                        "source": meta.get("source", "unknown"),
                        "distance": float(distance),
                        "relevance": _score_relevance(question, content, meta.get("company", company)),
                    }
                )

        if not all_docs:
            return {"retrieved_docs": []}

        ranked_docs = sorted(
            all_docs,
            key=lambda item: (item["relevance"] + max(0.0, 1.0 - item["distance"]), item["relevance"]),
            reverse=True,
        )
        return {"retrieved_docs": [
            {
                "content": doc["content"],
                "company": doc["company"],
                "source": doc["source"],
                "distance": doc["distance"],
            }
            for doc in ranked_docs[:4]
        ]}

    def generate_node(state: GraphState) -> dict[str, Any]:
        question = state.get("current_question", "")
        retrieved_docs = state.get("retrieved_docs", [])
        history = state.get("messages", [])

        if not retrieved_docs:
            answer = (
                "I could not find any authorized documents relevant to that question. "
                "Your account is not authorized to access documents for that company or topic."
            )
        else:
            context = "\n\n".join(f"Source ({doc['company']}): {doc['content']}" for doc in retrieved_docs[:4])
            history_text = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in history[-6:])
            prompt = f"""You are a helpful assistant answering questions from company earnings-call documents.
Use only the retrieved context below. If the context does not contain the answer, say so clearly.

Conversation history:
{history_text}

Retrieved context:
{context}

User question: {question}
"""
            if llm is None:
                answer = _build_fallback_answer(question, retrieved_docs)
            else:
                try:
                    response = llm.invoke(
                        [
                            SystemMessage(content="Answer briefly and strictly from the provided context. If the context is insufficient, say so clearly."),
                            HumanMessage(content=prompt),
                        ]
                    )
                    answer = _extract_text_from_response(response)
                    if not answer:
                        answer = _build_fallback_answer(question, retrieved_docs)
                except Exception:
                    answer = _build_fallback_answer(question, retrieved_docs)

        updated_history = list(history) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {"answer": answer, "messages": updated_history}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("generate_node", generate_node)
    workflow.set_entry_point("retrieve_node")
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", END)
    return workflow.compile(checkpointer=MemorySaver())


def run_graph(graph: StateGraph, thread_id: str, state: GraphState) -> dict[str, Any]:
    return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
