import pytest

from financial_risk.investigation.copilot import build_copilot_context, RetrievalResult
from financial_risk.investigation.llm_adapter import (
    build_evidence_only_fallback,
    run_grounded_copilot,
)


class FakeGenerator:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_grounded_copilot_adapter():
    context = build_copilot_context(
        "TXN42",
        [{"field": "amount", "value": 900, "signal": "amount", "severity": "high"}],
        [RetrievalResult("POL-1", 0.9, "Review unusual transaction amounts.")],
    )
    generator = FakeGenerator("Evidence-based investigation summary")
    result = run_grounded_copilot(context, generator)

    assert result.case_id == "TXN42"
    assert result.grounded is True
    assert result.response == "Evidence-based investigation summary"
    assert len(generator.prompts) == 1
    assert "Do not invent facts" in generator.prompts[0]


def test_copilot_requires_nonempty_generator_response():
    context = build_copilot_context("TXN42", [], [])
    with pytest.raises(ValueError, match="non-empty string"):
        run_grounded_copilot(context, FakeGenerator(" "))


def test_evidence_only_fallback():
    context = build_copilot_context(
        "TXN42",
        [{"field": "country", "value": "GB"}],
        [],
    )
    fallback = build_evidence_only_fallback(context)
    assert "TXN42" in fallback
    assert "country=GB" in fallback
    assert "No unsupported conclusions" in fallback
