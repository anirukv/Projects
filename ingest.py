from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover - optional dependency path
    GoogleGenerativeAIEmbeddings = None

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = "earnings_calls"
EMBEDDING_MODEL = "gemini-embedding-2"
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


def _require_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Create one at https://aistudio.google.com/apikey and set it in your environment or .env file."
        )
    return api_key


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


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


def _build_embeddings(texts: list[str]) -> list[list[float]]:
    if GoogleGenerativeAIEmbeddings is not None:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=_require_api_key(),
            )
            return embeddings.embed_documents(texts)
        except Exception as exc:  # pragma: no cover - runtime API fallback
            print(f"Falling back to deterministic embeddings because Gemini embedding failed: {exc}")

    return [_build_embedding(text) for text in texts]


def _infer_company_label(pdf_path: Path) -> str | None:
    stem = pdf_path.stem.lower()
    return COMPANY_ALIASES.get(stem)


def ingest_documents() -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"No docs directory found at {DOCS_DIR}")

    documents: list[dict[str, Any]] = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        company_name = _infer_company_label(pdf_path)
        if not company_name:
            continue

        text = _extract_text(pdf_path)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(
                {
                    "page_content": chunk,
                    "metadata": {
                        "company": company_name,
                        "source": pdf_path.name,
                    },
                }
            )

    if not documents:
        raise ValueError(f"No recognized company PDFs found in {DOCS_DIR}")

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # The ingestion step is safe to re-run: the collection is reused and new embeddings are
    # added for each chunk rather than duplicating content in the persistent Chroma store.
    texts = [item["page_content"] for item in documents]
    metadatas = [item["metadata"] for item in documents]
    embeddings_list = _build_embeddings(texts)
    ids = [f"{item['metadata']['company']}::{item['metadata']['source']}::{index}" for index, item in enumerate(documents)]

    collection.add(ids=ids, embeddings=embeddings_list, documents=texts, metadatas=metadatas)
    return len(documents)


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Indexed {count} chunks into {CHROMA_DB_DIR}")
