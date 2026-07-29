"""Document loading and text chunking utilities."""

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_COMPANY_MAP


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(documents_dir: Path) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for filename, company_id in DOCUMENT_COMPANY_MAP.items():
        path = documents_dir / filename
        if not path.exists():
            continue

        text = load_pdf_text(path)
        for index, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{filename}::chunk_{index}",
                    "filename": filename,
                    "company_id": company_id,
                    "text": chunk,
                }
            )

    return documents
