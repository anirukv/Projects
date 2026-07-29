"""TF-IDF-backed vector store with company-level access filtering."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K, VECTOR_STORE_DIR
from rag.document_loader import load_documents


class VectorStore:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.metadata: List[Dict[str, str]] = []

    def build_from_documents(self, documents_dir: Path) -> int:
        documents = load_documents(documents_dir)
        if not documents:
            raise ValueError(f"No documents found in {documents_dir}")

        texts = [doc["text"] for doc in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        self.metadata = documents
        return len(documents)

    def save(self, store_dir: Path) -> None:
        if self.vectorizer is None or self.matrix is None:
            raise ValueError("Vector store has not been built yet.")

        store_dir.mkdir(parents=True, exist_ok=True)
        with open(store_dir / "vectorizer.pkl", "wb") as handle:
            pickle.dump(self.vectorizer, handle)
        with open(store_dir / "matrix.pkl", "wb") as handle:
            pickle.dump(self.matrix, handle)
        with open(store_dir / "metadata.pkl", "wb") as handle:
            pickle.dump(self.metadata, handle)
        with open(store_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump({"backend": "tfidf"}, handle)

    def load(self, store_dir: Path) -> None:
        vectorizer_path = store_dir / "vectorizer.pkl"
        matrix_path = store_dir / "matrix.pkl"
        metadata_path = store_dir / "metadata.pkl"
        if not all(path.exists() for path in (vectorizer_path, matrix_path, metadata_path)):
            raise FileNotFoundError(f"Vector store not found in {store_dir}")

        with open(vectorizer_path, "rb") as handle:
            self.vectorizer = pickle.load(handle)
        with open(matrix_path, "rb") as handle:
            self.matrix = pickle.load(handle)
        with open(metadata_path, "rb") as handle:
            self.metadata = pickle.load(handle)

    def search(
        self,
        query: str,
        allowed_companies: Set[str],
        top_k: int = TOP_K,
    ) -> List[Dict[str, str]]:
        if self.vectorizer is None or self.matrix is None or not self.metadata:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked_indices = np.argsort(scores)[::-1]
        results: List[Dict[str, str]] = []

        for idx in ranked_indices:
            doc = self.metadata[idx]
            if doc["company_id"] not in allowed_companies:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                {
                    "score": score,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "company_id": doc["company_id"],
                }
            )
            if len(results) >= top_k:
                break

        return results
