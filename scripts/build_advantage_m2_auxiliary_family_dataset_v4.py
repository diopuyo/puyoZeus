"""正本M2行へ旧308列由来の因果的auxiliary family教師だけを結合する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import build_advantage_m2_dataset_v1 as base
from scripts import train_advantage_m2_auxiliary_v1 as m2_train
from scripts import train_advantage_m2_old308_anchor_ledger_v1 as old_train
from src.advantage_m2_bounded_auxiliary_v4 import (
    AUXILIARY_FAMILY_NAMES,
    AUXILIARY_FEATURE_NAMES_V4,
    MATERIAL_SPATIAL_AUXILIARY_NAMES,
)


FORMAT_VERSION = "advantage-m2-auxiliary-family-dataset/v4"
COMPLETE_VERSION = "advantage-m2-auxiliary-family-dataset-complete/v4"
DEFAULT_M2_ROOT = Path(
    "data/verify/advantage_m2_canonical_dataset_46v_2026-09-05_v1_material_spatial8"
)
DEFAULT_OLD308_ROOT = Path(
    "data/verify/advantage_m2_old308_anchor_dataset_46v_2026-09-05_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/advantage_m2_auxiliary_family_dataset_46v_2026-09-06_v4"
)


class AdvantageM2AuxiliaryFamilyDatasetError(RuntimeError):
    """auxiliary family結合の入力・receipt契約違反。"""


def _validate_identity(m2: m2_train.M2SamplesV1, old: old_train.Samples) -> None:
    pairs = (
        (m2.state_ids, old.state_ids, "state_id"),
        (m2.source_groups, old.groups, "source"),
        (m2.game_keys, old.games, "game"),
        (m2.labels, old.labels, "label"),
        (m2.weights, old.weights, "weight"),
        (m2.folds, old.folds, "fold"),
    )
    for left, right, name in pairs:
        if not np.array_equal(left, right):
            raise AdvantageM2AuxiliaryFamilyDatasetError(f"M2/old308 {name}が不一致です")


def _old_feature_columns(old: old_train.Samples) -> dict[str, int]:
    columns = {name: index for index, name in enumerate(old.feature_names)}
    if len(columns) != len(old.feature_names):
        raise AdvantageM2AuxiliaryFamilyDatasetError("old308 feature名が重複しています")
    return columns


def _extract_old_feature(
    old: old_train.Samples, columns: Mapping[str, int], name: str,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"a_indicator_{name}"
    required = tuple(
        f"{prefix}_{suffix}" for suffix in ("p1", "p2", "p1_missing", "p2_missing")
    )
    if any(item not in columns for item in required):
        raise AdvantageM2AuxiliaryFamilyDatasetError(f"old308列が不足しています: {name}")
    values = np.stack((
        old.features[:, columns[required[0]]], old.features[:, columns[required[1]]],
    ), axis=1).astype(np.float32)
    missing = np.stack((
        old.features[:, columns[required[2]]], old.features[:, columns[required[3]]],
    ), axis=1)
    if not np.logical_or(missing == 0.0, missing == 1.0).all():
        raise AdvantageM2AuxiliaryFamilyDatasetError(f"missing maskが0/1ではありません: {name}")
    mask = missing == 0.0
    values[~mask] = 0.0
    return values, mask.astype(np.bool_)


def build_targets(
    m2: m2_train.M2SamplesV1, old: old_train.Samples,
) -> tuple[np.ndarray, np.ndarray]:
    """既存8教師を保持し、新family教師をold308 side列から抽出する。"""

    count, width = len(m2.labels), len(AUXILIARY_FEATURE_NAMES_V4)
    values = np.zeros((count, 2, width), dtype=np.float32)
    mask = np.zeros((count, 2, width), dtype=np.bool_)
    existing = len(MATERIAL_SPATIAL_AUXILIARY_NAMES)
    values[..., :existing] = m2.auxiliary_targets
    mask[..., :existing] = m2.auxiliary_mask
    columns = _old_feature_columns(old)
    for index, name in enumerate(AUXILIARY_FEATURE_NAMES_V4[existing:], start=existing):
        values[..., index], mask[..., index] = _extract_old_feature(old, columns, name)
    _validate_targets(values, mask)
    return values, mask


def _validate_targets(values: np.ndarray, mask: np.ndarray) -> None:
    if values.dtype != np.float32 or mask.dtype != np.bool_ or values.shape != mask.shape:
        raise AdvantageM2AuxiliaryFamilyDatasetError("auxiliary shape/dtypeが不正です")
    selected = values[mask]
    if not np.isfinite(selected).all() or np.any((selected < 0.0) | (selected > 1.0)):
        raise AdvantageM2AuxiliaryFamilyDatasetError("auxiliary教師は有限な0..1必須です")


def build(args: argparse.Namespace) -> dict[str, Any]:
    """検証済み親2系統を結合し、新規rootへ排他的に保存する。"""

    if args.output_root.exists():
        raise AdvantageM2AuxiliaryFamilyDatasetError(f"出力先は新規必須です: {args.output_root}")
    m2, m2_manifest = m2_train.load_m2_dataset(args.m2_root)
    old, old_manifest = old_train.load_dataset(args.old308_root)
    _validate_identity(m2, old)
    targets, masks = build_targets(m2, old)
    arrays = {"state_ids": m2.state_ids, "auxiliary_targets": targets,
              "auxiliary_mask": masks}
    args.output_root.mkdir(parents=True, exist_ok=False)
    data_path = args.output_root / "dataset.npz"
    np.savez_compressed(data_path, **arrays)
    manifest = _manifest(args, m2, m2_manifest, old_manifest, arrays, data_path)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
    )
    complete = _complete(manifest_path, data_path)
    (args.output_root / "COMPLETE").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
    )
    return complete


def _manifest(
    args: argparse.Namespace, samples: m2_train.M2SamplesV1,
    m2_manifest: Mapping[str, Any], old_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray], data_path: Path,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "state_count": len(samples.labels),
        "source_count": len(np.unique(samples.source_groups)),
        "game_count": len(np.unique(samples.game_keys)),
        "parent_m2_root": str(args.m2_root.resolve()),
        "parent_m2_manifest_sha256": base.file_sha256(args.m2_root / "manifest.json"),
        "parent_m2_dataset_sha256": m2_manifest["dataset"]["sha256"],
        "old308_root": str(args.old308_root.resolve()),
        "old308_manifest_sha256": base.file_sha256(args.old308_root / "manifest.json"),
        "old308_dataset_sha256": old_manifest["dataset"]["sha256"],
        "feature_order": list(AUXILIARY_FEATURE_NAMES_V4),
        "families": {name: list(values) for name, values in AUXILIARY_FAMILY_NAMES.items()},
        "auxiliary_role": "training_only_side_head_not_win_probability_input",
        "shortcuts_excluded": ["score", "tsumo_count", "recognition_quality", "mechanism"],
        "array_sha256": {name: base._array_sha256(value) for name, value in arrays.items()},
        "dataset": {"name": data_path.name, "sha256": base.file_sha256(data_path)},
        "production_config_changed": False,
        "attack_difference_ten_percent_correction": False,
        "code_sha256": {
            "builder": base.file_sha256(Path(__file__)),
            "model": base.file_sha256(
                Path(__file__).resolve().parents[1] / "src/advantage_m2_bounded_auxiliary_v4.py"
            ),
            "production_config": base.file_sha256(
                Path(__file__).resolve().parents[1] / "src/production_config.py"
            ),
        },
    }


def _complete(manifest_path: Path, data_path: Path) -> dict[str, Any]:
    return {
        "format_version": COMPLETE_VERSION, "not_production": True,
        "manifest_sha256": base.file_sha256(manifest_path),
        "dataset_sha256": base.file_sha256(data_path),
        "production_config_changed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--old308-root", type=Path, default=DEFAULT_OLD308_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = build(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
