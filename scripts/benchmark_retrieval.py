"""Small authored retrieval regression benchmark, not an independent quality study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_risk.investigation.rag import load_chunks
from financial_risk.investigation.retrieval import (
    MODEL, REVISION, LexicalRetriever, SemanticRetriever, SentenceEncoder,
)


def evaluate(engine, questions):
    rows = []
    for q in questions:
        started = time.perf_counter()
        hits = engine.search(q["query"], top_k=3)
        elapsed = (time.perf_counter() - started) * 1000
        ids = [h.document_id for h in hits]
        relevant = set(q["relevant"])
        ranks = [i + 1 for i, value in enumerate(ids) if value in relevant]
        rows.append({"id": q["id"], "query": q["query"], "retrieved": ids,
                     "scores": [h.score for h in hits], "latency_ms": elapsed,
                     "recall_at_3": len(relevant.intersection(ids)) / len(relevant)
                     if relevant else None,
                     "reciprocal_rank_at_3": 1 / min(ranks) if ranks else 0,
                     "out_of_domain": not relevant,
                     "accepted": any(h.score >= engine.minimum_score for h in hits)})
    answerable = [r for r in rows if not r["out_of_domain"]]
    negatives = [r for r in rows if r["out_of_domain"]]
    return {"recall_at_3": float(np.mean([r["recall_at_3"] for r in answerable])),
            "mrr_at_3": float(np.mean([r["reciprocal_rank_at_3"] for r in answerable])),
            "ood_false_accept_rate": float(np.mean([r["accepted"] for r in negatives])),
            "p50_query_ms": float(np.median([r["latency_ms"] for r in rows])),
            "p95_query_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "threshold": engine.minimum_score, "questions": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval-benchmark.json"))
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/retrieval_questions.json"
    questions = json.loads(fixture.read_text())
    docs = load_chunks(ROOT)
    known_ids = set(docs["document_id"])
    if any(not set(q["relevant"]).issubset(known_ids) for q in questions):
        raise ValueError("Relevance labels reference missing chunks")
    engines = {"tfidf": LexicalRetriever(docs)}
    started = time.perf_counter()
    if args.semantic:
        engines["semantic"] = SemanticRetriever(
            docs, SentenceEncoder(allow_download=args.allow_download))
    startup = time.perf_counter() - started
    result = {"description": "13 authored regression questions, 6 chunks; not held-out evidence",
              "corpus_sha256": hashlib.sha256(docs.to_json().encode()).hexdigest(),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "python": platform.python_version(), "platform": platform.platform(),
              "model": MODEL if args.semantic else None,
              "revision": REVISION if args.semantic else None,
              "semantic_startup_seconds": startup if args.semantic else None,
              "versions": {p: importlib.metadata.version(p) for p in
                           (["numpy", "scikit-learn", "sentence-transformers", "torch"]
                            if args.semantic else ["numpy", "scikit-learn"])},
              "results": {name: evaluate(engine, questions) for name, engine in engines.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, metrics in result["results"].items():
        print(name, {k: v for k, v in metrics.items() if k != "questions"})


if __name__ == "__main__":
    main()
