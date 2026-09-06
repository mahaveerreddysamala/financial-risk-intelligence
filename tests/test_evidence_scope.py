import pandas as pd
import pytest

from financial_risk.investigation.evidence_scope import (
    EvidenceScope, REFERENCE_SCOPE, required_fact, unavailable_evidence,
)
from financial_risk.investigation.rag import answer_question


@pytest.mark.parametrize("question,kind", [
    ("What is the exact financial loss for transaction TX-999?", "transaction_loss"),
    ("Please identify the owner of this account.", "customer_identity"),
    ("Predict the exact fraud rate at our company next quarter.", "organization_forecast"),
    ("Ｗｈｏ owns this account?", "customer_identity"),
])
def test_missing_fact_types(question, kind):
    assert required_fact(question) == kind
    assert unavailable_evidence(question, REFERENCE_SCOPE)
    assert unavailable_evidence(question, EvidenceScope(frozenset({kind}))) is None


@pytest.mark.parametrize("question", [
    "How do shared devices affect investigation?",
    "Does a high score prove that a customer committed fraud?",
    "What is the fraud model probability for this case?",
    "Should the current transaction be excluded from historical features?",
    "Can a forecast guarantee future performance?",
])
def test_general_guidance_and_allowed_signals_are_not_gated(question):
    assert unavailable_evidence(question, REFERENCE_SCOPE) is None


def test_gate_precedes_retrieval_and_generation_even_with_case_context():
    class NeverCalled:
        minimum_score = 0.35

        def search(self, *args, **kwargs):
            raise AssertionError("retrieval must not run")

        def generate(self, *args, **kwargs):
            raise AssertionError("provider must not run")

    answer = answer_question("What was the actual loss for this transaction?", pd.DataFrame(),
                             generator=NeverCalled(), retriever=NeverCalled(),
                             case={"evidence": [{"field": "fraud_probability", "value": 0.9}]},
                             evidence_scope=REFERENCE_SCOPE)
    assert answer.abstained and answer.mode == "evidence-unavailable"
    assert not answer.sources and not answer.citation_ids_valid


def test_custom_corpus_is_not_implicitly_declared_reference_only():
    docs = pd.DataFrame({"document_id": ["loss"], "text": ["Actual transaction loss: 25"]})
    answer = answer_question("What is the actual transaction loss?", docs)
    assert not answer.abstained
