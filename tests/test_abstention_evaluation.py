import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "abstention", Path(__file__).resolve().parents[1] / "scripts/evaluate_abstention.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(query, unsupported, value):
    return {"query": query, "out_of_domain": unsupported, "scores": [value]}


def test_validation_does_not_select_threshold_and_regression_fails_gate():
    dev = [row("dev yes", False, 0.38), row("dev no", True, 0.355)]
    val = [row("new yes", False, 0.355), row("new no", True, 0.355)]
    result = module.compare(dev, val)
    assert result["selected_threshold"] == 0.36
    assert result["validation_candidate"]["false_rejects"] == 1
    assert not result["meets_predeclared_gate"]
    val[0]["scores"] = [0.9]
    assert module.compare(dev, val)["meets_predeclared_gate"]


def test_overlap_and_missing_class_rejected():
    rows = [row("same QUESTION", False, 0.8), row("negative", True, 0.1)]
    with pytest.raises(ValueError, match="overlap"):
        module.compare(rows, [row(" Same question ", False, 0.9), row("other", True, 0.1)])
    with pytest.raises(ValueError, match="Both"):
        module.score(rows[:1], 0.35)


def test_threshold_equality_is_accepted():
    result = module.score([row("yes", False, 0.35), row("no", True, 0.35)], 0.35)
    assert result["false_accepts"] == 1
    assert result["false_rejects"] == 0
