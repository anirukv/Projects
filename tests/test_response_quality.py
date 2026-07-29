from graph import _build_fallback_answer


def test_fallback_answer_is_concise_and_grounded():
    docs = [
        {
            "content": "Google reported strong growth in advertising revenue and operating margins last year. The company also expanded its product portfolio.",
            "company": "CompanyA",
            "source": "Google.pdf",
            "distance": 0.2,
            "relevance": 4.0,
        }
    ]

    answer = _build_fallback_answer("What does Google say about its business performance?", docs)

    assert "CompanyA" in answer
    assert "Google" in answer
    assert "advertising revenue" in answer or "operating margins" in answer
    assert len(answer) < 500
