"""Evidence-grounded retrieval and prompt construction for investigations."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    score: float
    text: str


@dataclass(frozen=True)
class CopilotContext:
    case_id: str
    evidence: tuple[dict, ...]
    retrieved_documents: tuple[RetrievalResult, ...]


def build_document_index(documents: pd.DataFrame) -> tuple[TfidfVectorizer, object, list[str]]:
    """Build a lightweight TF-IDF retrieval index for policy/typology documents."""
    required = {"document_id", "text"}
    missing = required.difference(documents.columns)
    if missing:
        raise ValueError(f"Missing document fields: {sorted(missing)}")
    if documents.empty:
        raise ValueError("documents must not be empty")
    texts = documents["text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, documents["document_id"].astype(str).tolist()


def retrieve_documents(
    query: str,
    documents: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: object,
    top_k: int = 3,
) -> list[RetrievalResult]:
    """Retrieve the most relevant policy/typology documents for a query."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(documents) != matrix.shape[0]:
        raise ValueError("documents and retrieval matrix must have the same row count")

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
    return [
        RetrievalResult(
            document_id=str(documents.iloc[i]["document_id"]),
            score=float(scores[i]),
            text=str(documents.iloc[i]["text"]),
        )
        for i in indices
    ]


def build_copilot_context(
    case_id: str,
    evidence: Iterable[dict],
    retrieved_documents: Iterable[RetrievalResult],
) -> CopilotContext:
    """Create an immutable context containing only supplied case evidence and retrievals."""
    return CopilotContext(
        case_id=str(case_id),
        evidence=tuple(dict(item) for item in evidence),
        retrieved_documents=tuple(retrieved_documents),
    )


def build_grounded_prompt(context: CopilotContext) -> str:
    """Build a prompt that requires the downstream LLM to stay within provided evidence."""
    evidence_lines = [
        f"- {item.get('field')}: {item.get('value')} (signal={item.get('signal')}, severity={item.get('severity')})"
        for item in context.evidence
    ]
    document_lines = [f"[{doc.document_id}] {doc.text}" for doc in context.retrieved_documents]
    return "\n".join(
        [
            "You are a financial investigation copilot.",
            "Use only the supplied case evidence and retrieved reference documents.",
            "Do not invent facts, entities, transactions, policy requirements, or motives.",
            "Clearly distinguish observed evidence from interpretation.",
            "If the evidence is insufficient, say that it is insufficient.",
            "Return: risk summary, evidence-based findings, policy/typology references, and recommended analyst next steps.",
            f"CASE_ID: {context.case_id}",
            "CASE EVIDENCE:",
            *(evidence_lines or ["- none supplied"]),
            "RETRIEVED REFERENCES:",
            *(document_lines or ["- none retrieved"]),
        ]
    )
