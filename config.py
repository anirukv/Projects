"""Application configuration: users, companies, and access control."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

COMPANIES = {
    "company_a": "Alpha Corp",
    "company_b": "Beta Industries",
    "company_c": "Gamma Holdings",
    "company_d": "Delta Systems",
    "company_e": "Epsilon Energy",
}

USERS = {
    "alice@email.com": {
        "name": "Alice",
        "companies": ["company_a"],
    },
    "bob@email.com": {
        "name": "Bob",
        "companies": ["company_b", "company_c"],
    },
    "charlie@email.com": {
        "name": "Charlie",
        "companies": ["company_d", "company_e"],
    },
}

DOCUMENT_COMPANY_MAP = {
    "company_a_earnings_q4.pdf": "company_a",
    "company_b_earnings_q4.pdf": "company_b",
    "company_c_earnings_q4.pdf": "company_c",
    "company_d_earnings_q4.pdf": "company_d",
    "company_e_earnings_q4.pdf": "company_e",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.08
