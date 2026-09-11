import json
from pathlib import Path

import httpx
import pytest

from financial_risk.investigation.demo import run_demo
from financial_risk.investigation.evaluate_generation import evaluate
from financial_risk.investigation.local_llm import OllamaTextGenerator
from financial_risk.investigation.rag import answer_question, load_chunks

ROOT = Path(__file__).parents[1]


def test_local_transport_and_usage():
    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(request.content)
        assert not payload["stream"] and not payload["think"]
        assert payload["options"]["num_predict"] == 600
        assert "ONLY allowed citation tokens" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"done": True, "done_reason": "stop",
            "message": {"content": "Shared devices require contextual review [S1]."},
            "prompt_eval_count": 20, "eval_count": 10})
    gen = OllamaTextGenerator(transport=httpx.MockTransport(handler))
    assert answer_question("shared devices", load_chunks(ROOT), gen).mode == "llm-unverified"
    assert gen.last_usage["total_tokens"] == 30


@pytest.mark.parametrize("body", [
    {"done": True, "done_reason": "length", "message": {"content": "Truncated [S1]"}},
    {"done": False, "message": {"content": "Partial [S1]"}},
    {"done": True, "message": {"content": " "}},
])
def test_incomplete_local_output_withheld(body):
    gen = OllamaTextGenerator(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    assert answer_question("shared devices", load_chunks(ROOT), gen).mode == "provider-error"


def test_local_failure_has_no_hosted_fallback():
    gen = OllamaTextGenerator(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    result = answer_question("shared devices", load_chunks(ROOT), gen)
    assert result.mode == "provider-error" and result.abstained


@pytest.mark.parametrize("model", ["", "model-cloud", "remote/model"])
def test_cloud_tags_rejected(model):
    with pytest.raises(ValueError):
        OllamaTextGenerator(model)


def test_exact_case_observations_are_separate_from_generation(tmp_path):
    class Generator:
        def generate(self, prompt):
            assert json.loads(prompt)["evidence"] == []
            return "Shared devices require contextual review [S1]."
    report = run_demo(ROOT, tmp_path, Generator())
    assert report["train_rows"] > 0 and report["test_rows"] > 0
    assert 0 <= report["case"]["risk_score"] <= 1
    assert report["exact_observations"][0]["field"] == "fraud_probability"
    assert not report["actions_executed"] and report["brief"]["citation_ids_valid"]
    assert (tmp_path / "analyst-brief.md").is_file()


def test_valid_citation_does_not_imply_groundedness():
    report = {"rows": [{"id": "q1", "provider_attempted": True, "citation_ids_valid": True,
                        "expected_behavior": "answer", "abstained": False, "latency_seconds": 2}]}
    assert evaluate(report)["quality"]["groundedness"]["pass_rate"] is None
    reviews = {"q1": {"groundedness": False, "citation_accuracy": False,
                      "reviewer": "test reviewer", "notes": "Claim contradicts passage"}}
    assert evaluate(report, reviews)["quality"]["groundedness"]["pass_rate"] == 0
    with pytest.raises(ValueError, match="Unknown"):
        evaluate(report, {"unknown": {}})
