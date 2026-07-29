"""Build and persist the vector index from PDF documents."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DOCUMENTS_DIR, VECTOR_STORE_DIR
from rag.vector_store import VectorStore

sys.path.insert(0, str(ROOT / "scripts"))
from generate_sample_pdfs import generate_sample_pdfs


def main() -> None:
    if not any(DOCUMENTS_DIR.glob("*.pdf")):
        print("Generating sample PDF documents...")
        generate_sample_pdfs()

    print("Building vector store...")
    store = VectorStore()
    count = store.build_from_documents(DOCUMENTS_DIR)
    store.save(VECTOR_STORE_DIR)
    print(f"Indexed {count} chunks into {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
