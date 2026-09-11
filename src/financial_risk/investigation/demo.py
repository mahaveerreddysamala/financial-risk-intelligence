"""Synthetic scoring to exact case observations and generated reference guidance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from financial_risk.dashboard.view_model import build_dashboard_snapshot, build_investigation_payload
from financial_risk.investigation.case_context import safe_case_sources
from financial_risk.investigation.evidence_scope import REFERENCE_SCOPE
from financial_risk.investigation.local_llm import OllamaTextGenerator
from financial_risk.investigation.rag import answer_question, load_chunks


def run_demo(root, output, generator=None, rows=2000):
    started = time.perf_counter()
    snapshot = build_dashboard_snapshot(rows=rows, seed=42)
    case = build_investigation_payload(snapshot.transactions.iloc[0])
    brief = answer_question(
        "Write three short sentences of reference-grounded review guidance about shared devices "
        "and model signals. Do not restate or interpret numeric case values: their calibrated "
        "meaning is unknown and exact observations will be displayed separately. "
        "Cite the reference passages and state that scores do not establish criminal intent.",
        load_chunks(root), generator, evidence_scope=REFERENCE_SCOPE,
    )
    report = {"synthetic": True, "seed": 42, "rows": rows,
              "train_rows": snapshot.train_rows, "test_rows": snapshot.test_rows,
              "selection": "highest combined risk score in synthetic holdout",
              "scoring": "XGBoost + IsolationForest + device/velocity heuristics; community proxy",
              "actions_executed": False, "case": case, "brief": asdict(brief),
              "exact_observations": safe_case_sources(case),
              "raw_generation_for_review": getattr(generator, "last_response", None),
              "model": getattr(generator, "model", None),
              "elapsed_seconds": round(time.perf_counter() - started, 3)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "demo.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    observations = "\n".join(f'- [{s["id"]}] {s["text"]}' for s in report["exact_observations"])
    (output / "analyst-brief.md").write_text(
        "# Synthetic analyst brief\n\n## Exact case observations\n\n" + observations
        + "\n\n## Generated reference guidance\n\n" + brief.text + "\n\n"
        "No transaction action was executed. Generated claims require review.\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/end-to-end-demo"))
    args = parser.parse_args()
    report = run_demo(Path.cwd(), args.output,
                      OllamaTextGenerator(args.model) if args.local else None, args.rows)
    print(f'Case {report["case"]["transaction_id"]}: {report["brief"]["mode"]}; {args.output}')
    if report["brief"]["abstained"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
