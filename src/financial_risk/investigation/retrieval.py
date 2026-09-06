"""Interchangeable lexical and local semantic retrievers with stable source IDs."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from financial_risk.investigation.copilot import (
    RetrievalResult, build_document_index, retrieve_documents,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class Retriever(Protocol):
    minimum_score: float

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]: ...


class LexicalRetriever:
    minimum_score = 0.12

    def __init__(self, documents: pd.DataFrame):
        self.documents = documents.copy(deep=True)
        self.vectorizer, self.matrix, _ = build_document_index(self.documents)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        return retrieve_documents(query, self.documents, self.vectorizer, self.matrix, top_k)


class SentenceEncoder:
    """Fixed local model; downloads require explicit opt-in, no remote code execution."""

    def __init__(self, *, allow_download: bool = False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            MODEL, revision=REVISION, device="cpu", trust_remote_code=False,
            local_files_only=not allow_download, model_kwargs={"use_safetensors": True},
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


def _normalize(values, rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError("Embedding shape must be (text count, positive dimension)")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero embeddings cannot be ranked")
    return matrix / norms


class SemanticRetriever:
    # Heuristic, deliberately separate from lexical scores; not calibrated confidence.
    minimum_score = 0.35

    def __init__(self, documents: pd.DataFrame, encoder):
        if documents.empty or not {"document_id", "text"}.issubset(documents.columns):
            raise ValueError("Nonempty documents with document_id and text are required")
        if documents["document_id"].duplicated().any():
            raise ValueError("Document IDs must be unique")
        self.documents = documents.copy(deep=True).reset_index(drop=True)
        self.encoder = encoder
        self.matrix = _normalize(encoder.encode(self.documents["text"].tolist()), len(documents))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip() or top_k < 1:
            raise ValueError("Nonempty query and positive top_k are required")
        vector = _normalize(self.encoder.encode([query]), 1)
        if vector.shape[1] != self.matrix.shape[1]:
            raise ValueError("Query and document embedding dimensions differ")
        scores = np.clip(self.matrix @ vector[0], -1, 1)
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [RetrievalResult(str(self.documents.iloc[i]["document_id"]),
                                float(scores[i]), str(self.documents.iloc[i]["text"]))
                for i in indices]
