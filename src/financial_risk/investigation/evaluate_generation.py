"""Separate mechanical answer checks from attributed claim reviews."""
import argparse
import json
import math
from pathlib import Path
import statistics


def evaluate(report, reviews=None):
    rows = report["rows"]
    generated = [r for r in rows if r["provider_attempted"]]
    abstentions = [r for r in rows if r["expected_behavior"] == "abstain"]
    answerable = [r for r in rows if r["expected_behavior"] == "answer"]
    times = sorted(r["latency_seconds"] for r in generated)
    reviews = reviews or {}
    if set(reviews) - {r["id"] for r in rows}:
        raise ValueError("Unknown review question IDs")
    result = {"generated_requests": len(generated),
              "citation_id_passes": sum(r["citation_ids_valid"] for r in generated),
              "expected_abstentions": len(abstentions),
              "correct_abstentions": sum(r["abstained"] for r in abstentions),
              "answerable_questions": len(answerable),
              "answerable_withheld": sum(r["abstained"] for r in answerable),
              "latency_p50_seconds": statistics.median(times) if times else None,
              "latency_p95_seconds": times[math.ceil(.95 * len(times)) - 1] if times else None,
              "quality": {}, "reviewers": sorted({v["reviewer"] for v in reviews.values()
                                                     if v.get("reviewer")}),
              "interpretation": "Citation IDs are syntax checks, not groundedness."}
    for dimension in ("groundedness", "citation_accuracy", "instruction_resistance"):
        scored = []
        for row in generated:
            review = reviews.get(row["id"], {})
            value = review.get(dimension)
            if value is not None:
                if type(value) is not bool or not review.get("reviewer") or not review.get("notes"):
                    raise ValueError("Reviews require boolean scores, reviewer and evidence notes")
                scored.append(value)
        result["quality"][dimension] = {"reviewed": len(scored), "passes": sum(scored),
                                        "pass_rate": sum(scored) / len(scored) if scored else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generation-evaluation.json"))
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()),
                      json.loads(args.reviews.read_text()) if args.reviews else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
