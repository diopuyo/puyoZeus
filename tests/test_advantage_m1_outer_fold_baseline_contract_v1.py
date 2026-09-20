"""二段目への全行cross-fitted baseline混入を防ぐ契約の回帰試験。"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from scripts import advantage_m1_outer_fold_baseline_contract_v1 as contract


def _record(fold: int) -> dict[str, Any]:
    tune = fold % len(contract.FOLD_IDS) + 1
    return {"eval_fold": fold, "tune_fold": tune,
            "train_folds": [value for value in contract.FOLD_IDS if value not in (fold, tune)]}


@pytest.mark.parametrize("fold", contract.FOLD_IDS)
def test_outer_checkpoint_contract_accepts_exact_fixed_plan(fold: int) -> None:
    records = {seed: _record(fold) for seed in contract.M1_SEEDS}
    contract.validate_outer_fold_seed_records(records, fold)


@pytest.mark.parametrize("fold", contract.FOLD_IDS)
def test_other_row_oof_model_cannot_be_reused_for_outer_training(fold: int) -> None:
    wrong = _record(fold % len(contract.FOLD_IDS) + 1)
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.validate_outer_fold_record(wrong, fold)


@pytest.mark.parametrize("field,value", [
    ("train_folds", [1, 4, 5, 6]), ("train_folds", [3, 4, 5, 5, 6]),
    ("train_folds", [3, 4, 5, 6.0]), ("tune_fold", 1), ("eval_fold", True),
])
def test_contaminated_or_ambiguous_provenance_is_rejected(field: str, value: Any) -> None:
    record = _record(1)
    record[field] = value
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.validate_outer_fold_record(record, 1)


def test_missing_seed_is_rejected() -> None:
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.validate_outer_fold_seed_records({contract.M1_SEEDS[0]: _record(1)}, 1)


def test_seed_average_is_float64_and_independent_of_mapping_order() -> None:
    values = {seed: np.array([0.1 + index * 0.1, 0.8], np.float32)
              for index, seed in enumerate(contract.M1_SEEDS)}
    reversed_values = dict(reversed(list(values.items())))
    result = contract.equal_seed_probability_mean(reversed_values)
    expected = np.mean(np.stack([values[seed].astype(np.float64) for seed in contract.M1_SEEDS]), axis=0)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([1.1]), np.array([[0.5]])])
def test_invalid_probability_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.equal_seed_probability_mean({seed: bad for seed in contract.M1_SEEDS})


def test_outer_oof_is_exact_and_non_eval_predictions_may_differ() -> None:
    stored = np.array([0.2, 0.7, 0.3])
    actual = np.array([0.2, 0.4, 0.3])
    assert contract.assert_outer_oof_exact(actual, stored, [1, 2, 1], 1) == 2
    changed = copy.copy(actual)
    changed[0] = np.nextafter(changed[0], 1.0)
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.assert_outer_oof_exact(changed, stored, [1, 2, 1], 1)


def test_empty_eval_comparison_is_rejected() -> None:
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.assert_outer_oof_exact(np.array([0.2]), np.array([0.2]), [2], 1)


@pytest.mark.parametrize("folds", [[True], [1.0]])
def test_fold_dtype_must_be_integer(folds: list[Any]) -> None:
    with pytest.raises(contract.OuterFoldBaselineError):
        contract.assert_outer_oof_exact(np.array([0.2]), np.array([0.2]), folds, 1)
