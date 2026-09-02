import pandas as pd
import pytest

from financial_risk.investigation.copilot import (
    build_copilot_context,
    build_document_index,
    build_grounded_prompt,
    retrieve_documents,
)


def _documents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": ["ATO-001", "VELOCITY-001", "MULE-001"],
            "text": [
                "Account takeover investigations should examine unusual device and location changes.",
                "Velocity fraud involves unusually frequent transactions over short time windows.",
                "Mule activity can involve devices or accounts shared across multiple customers.",
            ],
        }
    )


def test_retrieval_and_grounded_prompt():
    documents = _documents()
    vectorizer, matrix, ids = build_document_index(documents)
    assert len(ids) == 3

    results = retrieve_documents("rapid transaction velocity", documents, vectorizer, matrix, top_k=2)
    assert len(results) == 2
    assert results[0].document_id == "VELOCITY-001"
    assert results[0].score >= results[1].score

    context = build_copilot_context(
        "TXN123",
        [{"field": "txn_count_1h", "value": 9, "signal": "velocity_risk", "severity": "high"}],
        results,
    )
    prompt = build_grounded_prompt(context)
    assert "TXN123" in prompt
    assert "txn_count_1h" in prompt
    assert "Do not invent facts" in prompt
    assert "VELOCITY-001" in prompt


def test_copilot_validation():
    with pytest.raises(ValueError, match="must not be empty"):
        build_document_index(pd.DataFrame({"document_id": [], "text": []}))

    documents = _documents()
    vectorizer, matrix, _ = build_document_index(documents)
    with pytest.raises(ValueError, match="query"):
        retrieve_documents("", documents, vectorizer, matrix)
    with pytest.raises(ValueError, match="top_k"):
        retrieve_documents("velocity", documents, vectorizer, matrix, top_k=0)
    with pytest.raises(ValueError, match="same row count"):
        retrieve_documents("velocity", documents.iloc[:2], vectorizer, matrix)
