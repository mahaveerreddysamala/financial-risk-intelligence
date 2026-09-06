import importlib.util
from pathlib import Path

import pytest

from financial_risk.investigation.copilot import RetrievalResult

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_retrieval.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_ranking_and_acceptance_are_separate():
    class Engine:
        minimum_score = 0.35

        def search(self, query, top_k):
            return [RetrievalResult("wrong", 0.2, "<script>alert(1)</script>"),
                    RetrievalResult("right", 0.1, "evidence")]

    metrics = benchmark.evaluate(Engine(), [
        {"id": "a", "query": "<img src=x>", "relevant": ["right"]},
        {"id": "b", "query": "unsupported", "relevant": []},
    ])
    assert metrics["recall_at_3"] == 1
    assert metrics["mrr_at_3"] == pytest.approx(0.5)
    assert metrics["answerable_accept_rate"] == 0
    assert metrics["ood_false_accept_rate"] == 0
    report = benchmark.render_comparison({"results": {"test": metrics}})
    assert "<script>" not in report
    assert "<img src=x>" not in report
    assert "&lt;script&gt;" in report
    assert "abstained" in report
