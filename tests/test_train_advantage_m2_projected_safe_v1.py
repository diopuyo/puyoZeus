"""M2 projected-safe OOF runnerの契約試験。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_advantage_m2_projected_safe_v1 as trainer


def test_weighted_log_loss_uses_supplied_weights() -> None:
    labels = np.asarray([1.0, 0.0])
    probability = np.asarray([0.8, 0.4])
    weights = np.asarray([3.0, 1.0])
    expected = (-3.0 * np.log(0.8) - np.log(0.6)) / 4.0
    assert trainer.weighted_log_loss(labels, probability, weights) == pytest.approx(expected)


def test_weighted_log_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(trainer.AdvantageM2ProjectedSafeTrainingError, match="shape"):
        trainer.weighted_log_loss(np.ones(2), np.ones(3), np.ones(2))


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        ([0.8, 0.2, 0.5], [0.65, 0.35, 0.5]),
        ([0.8, 0.2], [0.8, 0.2]),
        ([0.6, 0.4], [0.55, 0.45]),
    ],
)
def test_bounded_probability_accepts_safe_shrink(
    baseline: list[float], candidate: list[float],
) -> None:
    trainer.validate_bounded_probability(np.asarray(baseline), np.asarray(candidate))


def test_bounded_probability_accepts_machine_rounding_at_boundary() -> None:
    trainer.validate_bounded_probability(
        np.asarray([0.8]), np.asarray([0.65 - 5e-13]),
    )


@pytest.mark.parametrize(
    "candidate",
    ([0.9, 0.1], [0.4, 0.6], [0.6, 0.4]),
)
def test_bounded_probability_rejects_expansion_crossing_or_excess(
    candidate: list[float],
) -> None:
    with pytest.raises(trainer.AdvantageM2ProjectedSafeTrainingError, match="safety"):
        trainer.validate_bounded_probability(
            np.asarray([0.8, 0.2]), np.asarray(candidate),
        )


def test_split_arrays_requires_supported_index() -> None:
    with pytest.raises(trainer.AdvantageM2ProjectedSafeTrainingError, match="不足"):
        trainer._split_arrays({})


def test_csv_ints_preserves_order() -> None:
    assert trainer._csv_ints("20260833,20260831") == (20260833, 20260831)


def test_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    with pytest.raises(trainer.AdvantageM2ProjectedSafeTrainingError, match="CUDA"):
        trainer._device("cuda")


def test_supported_fold_projection_uses_global_indices() -> None:
    samples = trainer.ProjectedSafeSamplesV1(
        full={"folds": np.asarray([1, 2, 3, 4], np.int8)},
        supported={"global_indices": np.asarray([1, 3], np.int32)},
    )
    np.testing.assert_array_equal(samples.supported_folds, np.asarray([2, 4], np.int8))


def test_initial_selection_requires_bit_identical_epoch_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = SimpleNamespace(
        full={"baseline_probability": np.asarray([0.8]), "folds": np.asarray([1]),
              "labels": np.asarray([1.0]), "weights": np.asarray([1.0])},
        global_indices=np.asarray([0]),
    )
    monkeypatch.setattr(trainer, "_predict_supported", lambda *_args: np.asarray([0.7]))
    with pytest.raises(trainer.AdvantageM2ProjectedSafeTrainingError, match="epoch 0"):
        trainer._initial_selection(object(), samples, np.asarray([0]), 1, object())
