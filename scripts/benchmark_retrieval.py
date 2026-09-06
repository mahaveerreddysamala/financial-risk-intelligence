"""Small authored retrieval regression benchmark, not an independent quality study."""
from __future__ import annotations

import argparse
import hashlib
import html
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
                     "scores": [h.score for h in hits], "texts": [h.text for h in hits],
                     "category": q.get("category", "original"), "latency_ms": elapsed,
                     "recall_at_3": len(relevant.intersection(ids)) / len(relevant)
                     if relevant else None,
                     "reciprocal_rank_at_3": 1 / min(ranks) if ranks else 0,
                     "out_of_domain": not relevant,
                     "accepted": any(h.score >= engine.minimum_score for h in hits)})
    answerable = [r for r in rows if not r["out_of_domain"]]
    negatives = [r for r in rows if r["out_of_domain"]]
    return {"answerable_count": len(answerable), "unsupported_count": len(negatives),
            "answerable_accept_rate": float(np.mean([r["accepted"] for r in answerable])),
            "recall_at_3": float(np.mean([r["recall_at_3"] for r in answerable])),
            "mrr_at_3": float(np.mean([r["reciprocal_rank_at_3"] for r in answerable])),
            "ood_false_accept_rate": float(np.mean([r["accepted"] for r in negatives])),
            "p50_query_ms": float(np.median([r["latency_ms"] for r in rows])),
            "p95_query_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "threshold": engine.minimum_score, "questions": rows}


def render_comparison(result):
    """Portable escaped HTML; includes excerpts and timings, never generated answers."""
    sections = []
    for name, metrics in result["results"].items():
        rows = []
        for q in metrics["questions"]:
            excerpts = "".join(f"<li>{html.escape(source)} ({score:.3f}): "
                               f"{html.escape(text)}</li>" for source, score, text in
                               zip(q["retrieved"], q["scores"], q["texts"]))
            rows.append(f'<details><summary>{html.escape(q["query"])} — '
                        f'{"accepted" if q["accepted"] else "abstained"}, '
                        f'{q["latency_ms"]:.2f} ms</summary><ul>{excerpts}</ul></details>')
        summary = {k: v for k, v in metrics.items() if k != "questions"}
        sections.append(f'<section><h2>{html.escape(name)}</h2><pre>'
                        f'{html.escape(json.dumps(summary, indent=2))}</pre>' + "".join(rows)
                        + '</section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Retrieval comparison</title><style>body{font:16px system-ui;'
            'max-width:1400px;margin:2rem auto;padding:1rem}main{display:flex;'
            'flex-wrap:wrap;gap:2rem}section{flex:1;min-width:280px}'
            'details{padding:12px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}'
            '</style><h1>Retrieval comparison</h1><p>AI-authored challenge questions; '
            'not independent quality evidence. Excerpts are not LLM answers. '
            'Acceptance uses heuristic thresholds. All top-three hits are shown, '
            'including hits below threshold.</p><main>' + "".join(sections) + '</main></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "tests/fixtures/retrieval_questions.json")
    parser.add_argument("--html", type=Path, help="Write an offline comparison report")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval-benchmark.json"))
    args = parser.parse_args()
    fixture = args.fixture
    questions = json.loads(fixture.read_text())
    docs = load_chunks(ROOT)
    known_ids = set(docs["document_id"])
    if (not questions or len({q["id"] for q in questions}) != len(questions)
            or any(not q["query"].strip() for q in questions)
            or not any(q["relevant"] for q in questions)
            or not any(not q["relevant"] for q in questions)):
        raise ValueError("Fixture needs unique IDs, nonempty queries and both answer classes")
    if any(not set(q["relevant"]).issubset(known_ids) for q in questions):
        raise ValueError("Relevance labels reference missing chunks")
    engines = {"tfidf": LexicalRetriever(docs)}
    started = time.perf_counter()
    if args.semantic:
        engines["semantic"] = SemanticRetriever(
            docs, SentenceEncoder(allow_download=args.allow_download))
    startup = time.perf_counter() - started
    result = {"description": f"{len(questions)} authored questions, {len(docs)} chunks; not held-out evidence",
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
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_comparison(result), encoding="utf-8")
    for name, metrics in result["results"].items():
        print(name, {k: v for k, v in metrics.items() if k != "questions"})


if __name__ == "__main__":
    main()
