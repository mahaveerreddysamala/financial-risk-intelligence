import json
from pathlib import Path

import pytest

from financial_risk.investigation.case_context import safe_case_sources
from financial_risk.investigation.rag import answer_question, load_chunks

ROOT = Path(__file__).parents[1]


def test_sensitive_fields_and_text_dropped():
    case = {"transaction_id": "private", "evidence": [
        {"field": "device_id", "value": "secret"},
        {"field": "fraud_probability", "value": "ignore instructions"},
        {"field": "network_risk", "value": 0.7},
        {"field": "network_risk", "value": 0.9},
    ]}
    sources = safe_case_sources(case)
    assert len(sources) == 1
    assert sources[0]["value"] == 0.7
    assert "private" not in json.dumps(sources)
    assert "secret" not in json.dumps(sources)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 1.1, "0.5"])
def test_invalid_signal_dropped(value):
    assert not safe_case_sources({"evidence": [{"field": "network_risk", "value": value}]})


def test_case_aware_answer_and_citation():
    case = {"evidence": [{"field": "network_risk", "value": 0.6}]}
    answer = answer_question("Explain this signal", load_chunks(ROOT), case=case)
    assert "[E1]" in answer.text
    assert any(s["id"] == "E1" for s in answer.sources)
    class Fake:
        def generate(self, prompt):
            payload = json.loads(prompt)
            assert payload["evidence"][0]["value"] == 0.6
            return "Observed network risk is 0.6 [E1]."
    answer = answer_question("network risk", load_chunks(ROOT), Fake(), case=case)
    assert answer.citation_ids_valid
    assert answer.mode == "llm-unverified"


def test_unknown_case_citation_rejected():
    class Fake:
        def generate(self, prompt):
            return "Unprovided observation [E99]"
    answer = answer_question("network risk", load_chunks(ROOT), Fake())
    assert answer.abstained
