import json
from pathlib import Path

import httpx
import pytest

from financial_risk.investigation.llm_pilot import LimitedGenerator, run_pilot
from financial_risk.investigation.rag import OpenAITextGenerator

ROOT = Path(__file__).parents[1]


def test_retrieval_pilot_never_claims_generation():
    report = run_pilot(ROOT)
    assert report["execution_mode"] == "retrieval-only"
    assert report["provider_attempts"] == 0
    assert not report["usage_complete"]
    assert all(row["manual_review"]["claims_supported"] is None for row in report["rows"])


def test_mock_transport_runs_full_pilot_and_reports_usage():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["max_completion_tokens"] == 800
        assert payload["store"] is False
        assert "untrusted" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content":
            "Review the supplied evidence with uncertainty [S1]."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

    generator = OpenAITextGenerator("fake-secret", "test-model", httpx.MockTransport(handler))
    report = run_pilot(ROOT, generator)
    assert report["provider_attempts"] == len(requests) == 4
    assert report["observed_total_tokens"] == 480
    assert report["usage_complete"]
    for row in report["rows"]:
        if row["expected_behavior"] == "abstain":
            assert row["abstained"] and not row["provider_attempted"]
    assert "fake-secret" not in json.dumps(report)
    assert report["factual_quality_status"].startswith("not evaluated")


def test_limits_count_failures_and_prevent_extra_network_requests():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, text="private-provider-detail")

    generator = OpenAITextGenerator("fake", "test", httpx.MockTransport(handler))
    report = run_pilot(ROOT, generator, max_requests=1)
    assert report["provider_attempts"] == len(requests) == 1
    assert report["usage_records"] == [None]
    assert not report["usage_complete"]
    assert "private-provider-detail" not in json.dumps(report)


def test_prompt_limit_applies_before_attempt_and_counts_utf8_bytes():
    class NeverCalled:
        def generate(self, prompt):
            pytest.fail("Oversized prompt reached generator")

    limited = LimitedGenerator(NeverCalled(), max_prompt_bytes=3)
    with pytest.raises(ValueError, match="prompt limit"):
        limited.generate("éé")
    assert limited.attempts == 0


def test_usage_does_not_leak_from_previous_success():
    responses = iter([
        httpx.Response(200, json={"choices": [{"message": {"content": "Answer [S1]"}}],
                                 "usage": {"total_tokens": 12}}),
        httpx.Response(503),
    ])
    generator = OpenAITextGenerator("fake", "test", httpx.MockTransport(lambda r: next(responses)))
    limited = LimitedGenerator(generator)
    limited.generate("one")
    with pytest.raises(httpx.HTTPError):
        limited.generate("two")
    assert limited.usage_records == [{"total_tokens": 12}, None]


def test_pilot_withholds_unknown_citations():
    class InvalidGenerator:
        def generate(self, prompt):
            return "Fabricated evidence [S999]"

    report = run_pilot(ROOT, InvalidGenerator())
    generated = [row for row in report["rows"] if row["provider_attempted"]]
    assert generated
    assert all(row["mode"] == "citation-rejected" for row in generated)
