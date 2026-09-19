"""M2 projected-safe-shrink事前登録監査の回帰テスト。"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts import analyze_advantage_m2_projected_safe_v1 as target


def _contract(**overrides: Any) -> dict[str, Any]:
    value = {
        "nonfinite_count": 0,
        "probability_expansion_count": 0,
        "even_crossing_count": 0,
        "unsupported_bit_mismatch_count": 0,
        "maximum_shrink_fraction": 0.5,
        "maximum_bound_excess": 0.0,
    }
    value.update(overrides)
    return value


def _comparison(delta: float = -0.01) -> dict[str, Any]:
    strata = {
        name: {"delta": {"log_loss": -0.001}}
        for name in target.REQUIRED_STRATA
    }
    return {
        "overall": {"delta": {"log_loss": delta, "brier": -0.001, "auc": 0.001}},
        "fold_win_count": 5,
        "source_bootstrap": {"ci95_high": -0.0001},
        "strata": strata,
        "extreme": {
            "0.1": {"new_count": 0, "candidate_nonincrease": True},
            "0.2": {"new_count": 0, "candidate_nonincrease": True},
        },
    }


def _plan() -> dict[str, Any]:
    return {
        "production_config_changed": False,
        "formal100_used": False,
        "hidden_reserve_used": False,
    }


def test_safety_contract_accepts_exact_half_shrink_and_fallback() -> None:
    baseline = np.asarray([0.1, 0.3, 0.5, 0.8], dtype=np.float64)
    candidate = np.asarray([0.3, 0.3, 0.5, 0.65], dtype=np.float64)

    contract = target._safety_contract(baseline, candidate, np.asarray([0, 3]))

    assert target._contract_passed(contract)
    assert contract["maximum_shrink_fraction"] == 0.5
    assert contract["unsupported_bit_mismatch_count"] == 0


def test_safety_contract_detects_unsupported_change_and_expansion() -> None:
    baseline = np.asarray([0.2, 0.8], dtype=np.float64)
    candidate = np.asarray([0.1, 0.7], dtype=np.float64)

    contract = target._safety_contract(baseline, candidate, np.asarray([0]))

    assert not target._contract_passed(contract)
    assert contract["probability_expansion_count"] == 1
    assert contract["unsupported_bit_mismatch_count"] == 1


def test_metric_gate_requires_five_fold_wins_and_stratum_noninferiority() -> None:
    comparison = _comparison()
    assert all(target._metric_gate(comparison).values())

    comparison["fold_win_count"] = 4
    comparison["strata"]["ledger_unusable"]["delta"]["log_loss"] = 1e-6
    checks = target._metric_gate(comparison)

    assert checks["fold_wins_at_least_5_of_6"] is False
    assert checks["stratum_ledger_unusable_noninferior"] is False


def test_gate_passes_only_when_metrics_safety_symmetry_and_isolation_pass() -> None:
    contracts = {str(seed): _contract() for seed in (1, 2, 3)}
    operational = [{
        "side_swap_max_absolute_error": 0.0,
        "saved_prediction_max_absolute_error": 0.0,
    }]

    gate = target._gate(_comparison(), contracts, operational, _plan())

    assert gate["passed"] is True
    contracts["2"]["even_crossing_count"] = 1
    assert target._gate(_comparison(), contracts, operational, _plan())["passed"] is False


def test_markdown_exposes_gate_failures() -> None:
    comparison = _comparison(delta=0.01)
    comparison["overall"]["delta"].update({"brier": 0.02, "auc": -0.03})
    comparison["source_bootstrap"].update({"ci95_low": -0.1})
    report = {
        "ensemble_comparison": comparison,
        "gate": {"passed": False, "checks": {"overall_log_loss_improved": False}},
    }

    markdown = target._markdown(report)

    assert "総合判定: **FAIL**" in markdown
    assert "- [ ] overall_log_loss_improved" in markdown
