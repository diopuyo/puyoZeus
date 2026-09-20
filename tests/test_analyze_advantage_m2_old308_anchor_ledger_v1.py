"""old308 ledger 3seed監査器の回帰テスト。"""

from __future__ import annotations

import numpy as np

from scripts import analyze_advantage_m2_old308_anchor_ledger_v1 as subject
from scripts import train_advantage_m2_old308_anchor_ledger_v1 as trainer


def _samples() -> trainer.Samples:
    count = 6
    return trainer.Samples(
        features=np.zeros((count, 308), dtype=np.float32),
        ledger_values=np.zeros((count, 2, 6), dtype=np.float32),
        ledger_availability=np.zeros((count, 2, 6, 5), dtype=np.float32),
        ledger_usable=np.asarray([True, True, False, True, False, True]),
        labels=np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8),
        weights=np.ones(count, dtype=np.float32),
        folds=np.arange(1, 7, dtype=np.int8),
        groups=np.asarray(["a", "a", "b", "b", "c", "c"]),
        games=np.asarray([f"game-{index}" for index in range(count)]),
        state_ids=np.asarray([f"state-{index}" for index in range(count)]),
        anchor_input_usable=np.ones(count, dtype=np.bool_),
        anchor_training_usable=np.ones(count, dtype=np.bool_),
        feature_names=tuple(f"feature-{index}" for index in range(308)),
    )


def test_ensemble_is_equal_probability_mean() -> None:
    predictions = {
        f"old308_anchor__seed_{seed}__calibrated": np.asarray([offset, 0.5])
        for seed, offset in zip(trainer.DEFAULT_SEEDS, (0.2, 0.3, 0.4), strict=True)
    }
    assert np.allclose(subject._ensemble(predictions, "old308_anchor"), [0.3, 0.5])


def test_extreme_audit_separates_new_and_resolved() -> None:
    samples = _samples()
    baseline = np.asarray([0.8, 0.8, 0.3, 0.7, 0.05, 0.95])
    candidate = np.asarray([0.05, 0.8, 0.8, 0.95, 0.8, 0.7])
    result = subject._extreme_audit(
        samples, baseline, candidate, np.ones(6, dtype=bool), np.arange(6),
    )
    assert result["0.1"]["new_count"] == 2
    assert result["0.1"]["resolved_count"] == 2
    assert result["0.1"]["candidate_nonincrease"] is True


def test_unsupported_audit_requires_bit_identical_anchor() -> None:
    samples = _samples()
    baseline = np.linspace(0.2, 0.7, 6)
    exact = baseline.copy()
    changed = baseline.copy()
    changed[2] += 0.01
    result = subject._unsupported_audit(
        samples, {"exact": exact, "changed": changed}, baseline,
    )
    assert result["exact"]["bit_mismatch_count"] == 0
    assert result["changed"]["bit_mismatch_count"] == 1


def test_gate_summary_enforces_all_numeric_gates() -> None:
    comparison = {
        "overall": {"delta": {"log_loss": -0.01, "brier": 0.0}},
        "source_bootstrap": {"ci95_high": -0.001},
        "fold_win_count": 5,
        "extreme": {
            "0.1": {"new_count": 0},
            "0.2": {"candidate_nonincrease": True},
        },
    }
    assert all(subject._gate_summary(comparison).values())


def test_feature_families_cover_each_registered_base_once() -> None:
    names = tuple(
        f"{base}_diff"
        for bases in subject.FEATURE_FAMILY_BASES.values() for base in bases
    )
    columns = subject._family_columns(names)
    assert sum(map(len, columns.values())) == len(names)
    assert set(columns) == set(subject.FEATURE_FAMILY_BASES)


def test_family_permutation_never_crosses_source() -> None:
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    order = subject._within_source_permutation(groups, seed=7)
    assert np.array_equal(groups[order], groups)
    assert sorted(order.tolist()) == list(range(len(groups)))
