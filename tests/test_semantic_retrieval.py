import numpy as np
import pandas as pd
import pytest

from financial_risk.investigation.retrieval import SemanticRetriever, _normalize
from financial_risk.investigation.rag import answer_question


class FakeEncoder:
    def encode(self, texts):
        return np.array([[1, 0] if t in {"device", "shared hardware"} else [0, 1]
                         for t in texts])


def corpus():
    return pd.DataFrame({"document_id": ["device-source", "other"],
                         "text": ["device", "velocity"]})


def test_semantic_ranking_and_rag_integration():
    docs = corpus()
    engine = SemanticRetriever(docs, FakeEncoder())
    docs.loc[0, "text"] = "changed after indexing"
    hits = engine.search("shared hardware")
    assert hits[0].document_id == "device-source"
    assert hits[0].score == 1
    answer = answer_question("shared hardware", docs, retriever=engine)
    assert answer.sources[0]["text"] == "device"
    assert len(answer.sources) == 1  # Zero-score hit is filtered.


@pytest.mark.parametrize("values,rows", [([[0, 0]], 1), ([[float("nan"), 1]], 1),
                                       ([1, 2], 1), ([[1, 2]], 2)])
def test_bad_embeddings_rejected(values, rows):
    with pytest.raises(ValueError):
        _normalize(values, rows)


def test_invalid_query_and_duplicate_ids():
    engine = SemanticRetriever(corpus(), FakeEncoder())
    with pytest.raises(ValueError):
        engine.search("")
    with pytest.raises(ValueError):
        engine.search("device", 0)
    docs = corpus()
    docs["document_id"] = "same"
    with pytest.raises(ValueError):
        SemanticRetriever(docs, FakeEncoder())


def test_cosine_normalization_and_stable_ties():
    class Encoder:
        def encode(self, texts):
            return [[3, 4]] * len(texts)
    hits = SemanticRetriever(corpus(), Encoder()).search("query")
    assert [h.document_id for h in hits] == ["device-source", "other"]
    assert all(h.score == pytest.approx(1) for h in hits)
