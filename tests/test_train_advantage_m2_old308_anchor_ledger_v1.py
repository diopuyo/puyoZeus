from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_advantage_m2_dataset_v1 as base
from scripts import build_advantage_m2_old308_anchor_dataset_v1 as builder
from scripts import train_advantage_m2_old308_anchor_ledger_v1 as subject


def test_random_control_deranges_each_non_singleton_stratum() -> None:
    values = np.arange(8 * 2 * 6, dtype=np.float32).reshape(8, 2, 6) / 100.0
    masks = np.zeros((8, 2, 6, 5), dtype=np.float32)
    masks[..., 0] = 1.0
    supported = np.ones(8, dtype=np.bool_)
    result = subject._randomize_ledger(values, masks, supported, seed=7)
    assert result.receipt["structural_deranged_count"] == 8
    assert result.receipt["singleton_count"] == 0
    assert result.receipt["unchanged_value_count"] == 0
    assert sorted(map(bytes, result.values.reshape(8, -1))) == sorted(
        map(bytes, values.reshape(8, -1))
    )


def test_random_control_disables_singleton_stratum() -> None:
    values = np.zeros((1, 2, 6), dtype=np.float32)
    masks = np.zeros((1, 2, 6, 5), dtype=np.float32)
    masks[..., 0] = 1.0
    result = subject._randomize_ledger(values, masks, np.ones(1, dtype=np.bool_), seed=1)
    assert result.supported.tolist() == [False]
    assert result.receipt["singleton_count"] == 1


def test_logit_clips_extreme_probabilities() -> None:
    output = subject._logit(np.asarray([0.0, 0.5, 1.0]))
    assert np.isfinite(output).all()
    assert output[0] < 0.0 < output[2]
    assert output[1] == pytest.approx(0.0)


def test_restore_unsupported_is_bit_identical_anchor() -> None:
    anchor = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    candidate = anchor + np.asarray([1e-8, 0.1, -1e-8])
    restored = subject._restore_unsupported(
        candidate, anchor, np.asarray([False, True, False]),
    )
    assert np.array_equal(restored[[0, 2]], anchor[[0, 2]])
    assert restored[1] == candidate[1]


def test_parse_args_rejects_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        subject.parse_args([
            "--output-root", str(tmp_path / "output"), "--variants", "unknown",
        ])


def test_load_dataset_validates_receipts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    arrays = _minimal_arrays()
    np.savez_compressed(root / "dataset.npz", **arrays)
    manifest = {
        "format_version": builder.FORMAT_VERSION,
        "not_production": True,
        "production_config_changed": False,
        "attack_difference_ten_percent_correction": False,
        "uses_337_or_338_feature_family": False,
        "weight_contract": {
            "source": "m2_canonical_dataset", "legacy_sample_weight_used": False,
        },
        "state_count": 6,
        "feature_names": [f"feature_{index}_diff" for index in range(308)],
        "array_sha256": {name: base._array_sha256(value) for name, value in arrays.items()},
        "dataset": {"sha256": base.file_sha256(root / "dataset.npz")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    complete = {
        "manifest_sha256": base.file_sha256(root / "manifest.json"),
        "dataset_sha256": base.file_sha256(root / "dataset.npz"),
    }
    (root / "COMPLETE").write_text(json.dumps(complete), encoding="utf-8")
    samples, loaded = subject.load_dataset(root)
    assert samples.features.shape == (6, 308)
    assert loaded["format_version"] == builder.FORMAT_VERSION
    arrays["labels"][0] = 1
    np.savez_compressed(root / "dataset.npz", **arrays)
    with pytest.raises(subject.Old308LedgerTrainingError, match="receipt"):
        subject.load_dataset(root)


def test_dataset_contract_rejects_legacy_weight_use() -> None:
    manifest = {
        "not_production": True,
        "production_config_changed": False,
        "attack_difference_ten_percent_correction": False,
        "uses_337_or_338_feature_family": False,
        "weight_contract": {
            "source": "m2_canonical_dataset", "legacy_sample_weight_used": True,
        },
    }
    with pytest.raises(subject.Old308LedgerTrainingError, match="監査専用"):
        subject._validate_dataset_contract(np.ones(2, dtype=np.float32), manifest)


def _minimal_arrays() -> dict[str, np.ndarray]:
    count = 6
    return {
        "old308_features": np.zeros((count, 308), dtype=np.float32),
        "ledger_values": np.zeros((count, 2, 6), dtype=np.float32),
        "ledger_availability": np.eye(5, dtype=np.float32)[
            np.zeros((count, 2, 6), dtype=np.int64)
        ],
        "ledger_usable": np.ones(count, dtype=np.bool_),
        "labels": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8),
        "weights": np.ones(count, dtype=np.float32),
        "folds": np.arange(1, 7, dtype=np.int8),
        "source_groups": np.asarray([f"source-{i}" for i in range(count)]),
        "game_keys": np.asarray([f"game-{i}" for i in range(count)]),
        "state_ids": np.asarray([f"state-{i}" for i in range(count)]),
        "anchor_input_usable": np.ones(count, dtype=np.bool_),
        "anchor_training_usable": np.ones(count, dtype=np.bool_),
    }
