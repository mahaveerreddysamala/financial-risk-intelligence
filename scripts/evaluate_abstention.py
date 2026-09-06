"""Evaluate a frozen development-selected threshold on separate authored questions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE = 0.35
CANDIDATES = (0.20, 0.25, 0.30, 0.35, 0.36, 0.40, 0.45, 0.50)


def score(rows, threshold):
    positives = [q for q in rows if not q["out_of_domain"]]
    negatives = [q for q in rows if q["out_of_domain"]]
    if not positives or not negatives:
        raise ValueError("Both answer classes are required")
    accepted = lambda q: any(s >= threshold for s in q["scores"])
    return {"threshold": threshold, "answerable_count": len(positives),
            "unsupported_count": len(negatives),
            "false_accepts": sum(accepted(q) for q in negatives),
            "false_rejects": sum(not accepted(q) for q in positives)}


def compare(development, validation):
    # Selection uses development only: minimize false accepts, then false rejects,
    # then choose the lower threshold. Candidate grid is fixed before validation.
    candidates = [score(development, t) for t in CANDIDATES]
    chosen = min(candidates, key=lambda s: (s["false_accepts"], s["false_rejects"],
                                           s["threshold"]))
    normalize = lambda q: " ".join(q["query"].lower().split())
    if {normalize(q) for q in development} & {normalize(q) for q in validation}:
        raise ValueError("Development and validation queries overlap")
    baseline = score(validation, BASELINE)
    candidate = score(validation, chosen["threshold"])
    improved = (candidate["false_accepts"] < baseline["false_accepts"]
                and candidate["false_rejects"] <= baseline["false_rejects"])
    return {"development_candidates": candidates, "selected_threshold": chosen["threshold"],
            "validation_baseline": baseline, "validation_candidate": candidate,
            "meets_predeclared_gate": improved,
            "gate": "Strictly fewer false accepts and no additional false rejects on validation",
            "deployment": "No runtime threshold changes; human review required even if gate passes",
            "limits": "AI-authored, same corpus; separate questions are not independent validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path,
                        default=Path("docs/retrieval-challenge-results.json"))
    parser.add_argument("--validation", type=Path,
                        default=Path("docs/retrieval-validation-results.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/abstention-experiment.json"))
    args = parser.parse_args()
    dev = json.loads(args.development.read_text())
    val = json.loads(args.validation.read_text())
    for key in ("model", "revision", "corpus_sha256"):
        if dev[key] != val[key]:
            raise ValueError(f"Reports differ in {key}")
    result = compare(dev["results"]["semantic"]["questions"],
                     val["results"]["semantic"]["questions"])
    result["report_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in (("development", args.development),
                                                  ("validation", args.validation))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
