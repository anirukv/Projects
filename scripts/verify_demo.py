"""Verify core Q&A functionality for demo users."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import USERS, VECTOR_STORE_DIR
from rag.qa_engine import QAEngine
from rag.vector_store import VectorStore


def run_demo() -> None:
    store = VectorStore()
    store.load(VECTOR_STORE_DIR)
    engine = QAEngine(store)

    tests = [
        ("alice@email.com", "What was the revenue reported in Q4?"),
        ("bob@email.com", "What was Beta Industries Q4 revenue?"),
        ("bob@email.com", "What about their product updates?"),
        ("charlie@email.com", "Tell me about Alpha Corp revenue"),
    ]

    print("=" * 60)
    print("Multi-User Document Search - Verification Demo")
    print("=" * 60)

    for email, question in tests:
        user = USERS[email]
        allowed = set(user["companies"])
        result = engine.ask(email, allowed, question)
        print(f"\nUser: {user['name']} ({email})")
        print(f"Question: {question}")
        print(f"Answer preview: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        if result["sources"]:
            source = result["sources"][0]
            print(f"Top source: {source['filename']} (score: {source['score']:.3f})")

    print("\n" + "=" * 60)
    print("Verification complete - environment is working.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
