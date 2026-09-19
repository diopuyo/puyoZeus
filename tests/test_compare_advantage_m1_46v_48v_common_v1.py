"""M1 46→48動画paired比較器の単体テスト。"""

from types import SimpleNamespace

import numpy as np

from scripts.compare_advantage_m1_46v_48v_common_v1 import (
    COMPARISON_ALIASES,
    COMPARISON_ROLES,
    _diagnostic_m0_exact_checks,
    _decision_policy,
    _gate_checks,
    _identity_audit,
)


def _samples(extra: bool = False) -> SimpleNamespace:
    count = 3 if extra else 2
    values = np.arange(count)
    return SimpleNamespace(
        boards=values[:, None, None, None].astype(np.int8),
        queues=values[:, None, None].astype(np.int8),
        ledger_values=np.zeros((count, 2, 6), dtype=np.float32),
        ledger_availability=np.zeros((count, 2, 6, 5), dtype=np.float32),
        labels=np.asarray([0.0, 1.0, 1.0][:count], dtype=np.float32),
        source_groups=np.asarray(["a", "b", "c"][:count]),
        game_keys=np.asarray(["ga", "gb", "gc"][:count]),
        weights=np.ones(count, dtype=np.float32),
        folds=np.asarray([1, 2, 3][:count], dtype=np.int8),
        state_ids=np.asarray(["s1", "s2", "s3"][:count]),
        available_ms=values.astype(np.int64),
        online_segment_index=values.astype(np.int32),
        ledger_usable=np.ones(count, dtype=bool),
        projected_input_digest=np.asarray(["d1", "d2", "d3"][:count]),
        projected_gate_status=np.asarray(["eligible"] * count),
    )


def _comparison(delta: float = -0.001) -> dict:
    severe = {"winner_below_10_percent_count": 0, "winner_below_20_percent_count": 1}
    return {
        "overall": {
            "candidate_minus_m0_log_loss": delta,
            "m0": {"auc": 0.6}, "candidate": {"auc": 0.61},
            "m0_severe": severe, "candidate_severe": severe,
        },
        "strata": {"incoming_at_least_30": {"candidate_minus_m0_log_loss": delta}},
        "fold_win_count": 6,
        "fold_rows": [{"candidate_minus_m0_log_loss": delta}] * 6,
        "source_cluster_bootstrap": {"probability_candidate_better": 0.99},
    }


def test_identity_audit_accepts_append_and_equal_relative_weights() -> None:
    old, new = _samples(), _samples(extra=True)
    positions, report = _identity_audit(old, new)
    assert positions.tolist() == [0, 1]
    assert report["all_exact_fields_pass"] is True
    assert report["weight_semantics"]["relative_weight_pass"] is True


def test_gate_checks_promotes_only_when_every_gate_passes() -> None:
    identity = {
        "all_exact_fields_pass": True,
        "weight_semantics": {
            "old_recomputed_bit_equal": True,
            "new_recomputed_bit_equal": True,
            "common_positive": True,
            "relative_weight_pass": True,
        },
        "production_config_unchanged": True,
    }
    extras = {"source_groups_pass": True, "game_key_collision_count": 0,
              "quarantine_contract_pass": True}
    comparisons = {"common_champion": _comparison(), "common_m0": _comparison(),
                   "full48_vs_m0": _comparison(), "full48_vs_random": _comparison()}
    result = _gate_checks(identity, extras, comparisons)
    assert result["all_pass"] is True
    assert result["decision"] == "PROMOTE_DEVELOPMENT_CHAMPION"
    comparisons["common_champion"] = _comparison(0.001)
    assert _gate_checks(identity, extras, comparisons)["decision"] == "KEEP_46V_CHAMPION"


def test_comparison_roles_disambiguate_legacy_m0_key() -> None:
    assert COMPARISON_ALIASES["common_primary"] == "common_champion"
    roles = COMPARISON_ROLES["common_champion"]
    assert roles["m0"] == "old_dataset_primary_variant_reference"
    assert roles["candidate"] == "new_dataset_primary_variant"


def test_diagnostic_policy_makes_promotion_impossible() -> None:
    gates = {"checks": {"all": True}, "all_pass": True,
             "decision": "PROMOTE_DEVELOPMENT_CHAMPION"}

    locked = _decision_policy(gates, diagnostic_only=True)

    assert locked["decision"] == "DIAGNOSTIC_ONLY"
    assert locked["standard_gate_decision"] == "PROMOTE_DEVELOPMENT_CHAMPION"
    assert locked["would_pass_standard_gate"] is True
    assert gates["decision"] == "PROMOTE_DEVELOPMENT_CHAMPION"


def test_diagnostic_policy_requires_common_m0_exact_identity() -> None:
    gates = {"checks": {"base": True}, "all_pass": True,
             "decision": "PROMOTE_DEVELOPMENT_CHAMPION"}
    common_m0 = _comparison(0.0)
    common_m0["overall"]["candidate"]["auc"] = 0.6
    comparisons = {"common_m0": common_m0}

    locked = _decision_policy(gates, True, comparisons)

    assert locked["diagnostic_integrity_pass"] is True
    assert locked["all_pass"] is True
    assert locked["checks"]["diagnostic_common_m0_log_loss_exact"] is True
    assert locked["checks"]["diagnostic_common_m0_auc_exact"] is True


def test_diagnostic_policy_fails_closed_on_common_m0_shift() -> None:
    gates = {"checks": {"base": True}, "all_pass": True,
             "decision": "PROMOTE_DEVELOPMENT_CHAMPION"}
    comparisons = {"common_m0": _comparison(1e-12)}

    locked = _decision_policy(gates, True, comparisons)

    assert locked["decision"] == "DIAGNOSTIC_ONLY"
    assert locked["diagnostic_integrity_pass"] is False
    assert locked["all_pass"] is False


def test_diagnostic_exact_checks_are_not_applied_without_comparisons() -> None:
    assert _diagnostic_m0_exact_checks(None) == {}
