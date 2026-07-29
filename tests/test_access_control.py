import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph import build_graph, create_initial_state, run_graph


def test_retrieval_is_restricted_to_allowed_companies():
    state = create_initial_state(
        user_email="charlie@email.com",
        allowed_companies=["Google"],
        messages=[],
    )
    state["current_question"] = "What are the financial highlights of Netflix?"
    graph = build_graph()
    result = run_graph(graph, "charlie-thread", state)
    assert result["retrieved_docs"] == []
