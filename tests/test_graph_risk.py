import pytest

from financial_risk.models.graph_risk import integrate_graph_risk


def test_graph_risk_preserves_base_score_without_community_signal() -> None:
    result = integrate_graph_risk(0.6, 0.2, 0.4, 0.1, 0.0)
    assert result.adjusted_score == pytest.approx(0.41)
    assert result.community_score == 0.0
    assert result.decision.level == "MEDIUM"


def test_community_signal_increases_risk_score() -> None:
    baseline = integrate_graph_risk(0.6, 0.2, 0.4, 0.1, 0.0)
    graph_aware = integrate_graph_risk(0.6, 0.2, 0.4, 0.1, 1.0)
    assert graph_aware.adjusted_score > baseline.adjusted_score
    assert graph_aware.adjusted_score == pytest.approx(0.46)


def test_graph_signal_can_change_operational_action() -> None:
    result = integrate_graph_risk(0.75, 0.4, 0.8, 0.5, 1.0)
    assert result.adjusted_score == pytest.approx(0.675)
    assert result.decision.action == "step_up_verification"


def test_invalid_signal_is_rejected() -> None:
    with pytest.raises(ValueError):
        integrate_graph_risk(0.6, 0.2, 1.2, 0.1, 0.0)


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        integrate_graph_risk(0.6, 0.2, 0.4, 0.1, 0.0, community_weight=0.2)
