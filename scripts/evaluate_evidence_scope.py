"""Compare offline retrieval acceptance with and without the declared-scope check."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_retrieval import evaluate
from financial_risk.investigation.evidence_scope import REFERENCE_SCOPE, unavailable_evidence
from financial_risk.investigation.rag import load_chunks
from financial_risk.investigation.retrieval import (
    MODEL, REVISION, LexicalRetriever, SemanticRetriever, SentenceEncoder,
)


def summarize(rows, scoped):
    def accepted(q):
        return q["accepted"] and not (scoped and q["scope_rejected"])
    positives = [q for q in rows if not q["out_of_domain"]]
    negatives = [q for q in rows if q["out_of_domain"]]
    return {"answerable_count": len(positives), "unsupported_count": len(negatives),
            "unsupported_accepted": sum(accepted(q) for q in negatives),
            "answerable_rejected": sum(not accepted(q) for q in positives),
            "scope_false_rejections": sum(q["scope_rejected"] for q in positives) if scoped else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="Use cached pinned MiniLM too")
    parser.add_argument("--output", type=Path, default=Path("docs/evidence-scope-results.json"))
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/evidence_scope_questions.json"
    questions = json.loads(fixture.read_text())
    normalize = lambda q: " ".join(q["query"].lower().split())
    previous = set()
    for name in ("retrieval_questions.json", "retrieval_challenge.json", "retrieval_validation.json"):
        previous.update(normalize(q) for q in json.loads((fixture.parent / name).read_text()))
    if previous & {normalize(q) for q in questions}:
        raise ValueError("New fixture overlaps prior queries")
    docs = load_chunks(ROOT)
    if any(not set(q["relevant"]).issubset(set(docs["document_id"])) for q in questions):
        raise ValueError("Unknown reference label")
    engines = {"tfidf": LexicalRetriever(docs)}
    if args.semantic:
        engines["semantic"] = SemanticRetriever(docs, SentenceEncoder())
    result = {"description": "36 AI-authored questions; same corpus, not independent evaluation",
              "python": platform.python_version(),
              "model": MODEL if args.semantic else None,
              "revision": REVISION if args.semantic else None,
              "versions": {p: importlib.metadata.version(p) for p in
                           (["numpy", "scikit-learn", "sentence-transformers", "torch"]
                            if args.semantic else ["numpy", "scikit-learn"])},
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "corpus_sha256": hashlib.sha256(docs.to_json().encode()).hexdigest(),
              "scope_code_sha256": hashlib.sha256((ROOT / "src/financial_risk/investigation/"
                                                    "evidence_scope.py").read_bytes()).hexdigest(),
              "results": {}}
    for name, engine in engines.items():
        report = evaluate(engine, questions)
        for q in report["questions"]:
            q["scope_rejected"] = unavailable_evidence(q["query"], REFERENCE_SCOPE) is not None
        result["results"][name] = {"baseline": summarize(report["questions"], False),
                                   "with_scope": summarize(report["questions"], True),
                                   "retrieval": report}
        print(name, {k: v for k, v in result["results"][name].items() if k != "retrieval"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
