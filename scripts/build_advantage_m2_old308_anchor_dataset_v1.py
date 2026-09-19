"""旧A_common_original 308列をM2 canonical行へ厳密結合する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import build_advantage_m2_dataset_v1 as m2_builder
from scripts import train_advantage_m2_auxiliary_v1 as m2_trainer
from src.event_indicator_features_v1 import FEATURE_TABLE_VERSION
from src.event_provisional_oof_v1 import MODEL_PARAMS


FORMAT_VERSION = "advantage-m2-old308-anchor-dataset/v1"
COMPLETE_VERSION = "advantage-m2-old308-anchor-dataset-complete/v1"
EXPECTED_FEATURE_COUNT = 308
EXPECTED_CANONICAL_SOURCE_COUNT = 46
EXPECTED_LEGACY_SOURCE_COUNT = 48
EXPECTED_STATE_COUNT = 224_011
DEFAULT_M2_ROOT = Path(
    "data/verify/advantage_m2_canonical_dataset_46v_2026-09-05_v1_material_spatial8"
)
DEFAULT_LEGACY_BASE = Path(
    "data/verify/indicator_redesign_48_inputs_online_v3_scope_uncertain_2026-08-31"
)
DEFAULT_FEATURE_ROOTS = (
    DEFAULT_LEGACY_BASE / "features_first30_part1_full",
    DEFAULT_LEGACY_BASE / "features_first30_part2_full",
    DEFAULT_LEGACY_BASE / "features_remaining18_full",
)
JOIN_COLUMNS = (
    "source_group_id", "available_ms", "online_segment_index",
)
AUDIT_COLUMNS = (
    "state_id", "game_key", "p1_won", "partition_fold", "sample_weight",
    "a_input_usable", "training_usable",
)
SHORTCUT_PATTERNS = {
    "score": ("_score_",),
    "progress": ("_tsumo_",),
    "recognition_quality": ("stable_confidence", "unknown_ratio"),
    "mechanism": ("_mechanism_",),
    "player_identity": ("player_identity",),
    "operation_speed": ("operation_speed",),
    "cumulative_match_score": ("cumulative_match_score",),
}


class Old308AnchorDatasetError(RuntimeError):
    """旧308列とM2 canonicalの結合契約に違反した。"""


@dataclass(frozen=True, slots=True)
class CanonicalJoinRows:
    """M2 canonicalから引き継ぐ同定・学習・ledger列。"""

    state_ids: np.ndarray
    source_groups: np.ndarray
    game_keys: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    available_ms: np.ndarray
    online_segment_index: np.ndarray
    ledger_values: np.ndarray
    ledger_availability: np.ndarray
    ledger_usable: np.ndarray


@dataclass(frozen=True, slots=True)
class LegacyFeatureSource:
    """hash検証済み旧特徴量1動画。"""

    target_id: str
    source_group_id: str
    partition_fold: int
    feature_path: Path
    feature_sha256: str
    manifest_path: Path
    manifest_sha256: str
    feature_names: tuple[str, ...]
    row_count: int


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Old308AnchorDatasetError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise Old308AnchorDatasetError(f"JSONはobject必須です: {path}")
    return value


def _load_canonical(root: Path) -> tuple[CanonicalJoinRows, dict[str, Any]]:
    samples, manifest = m2_trainer.load_m2_dataset(root)
    data_path = root / "dataset.npz"
    try:
        with np.load(data_path, allow_pickle=False) as data:
            available_ms = np.asarray(data["available_ms"], dtype=np.int64)
            segment = np.asarray(data["online_segment_index"], dtype=np.int32)
            ledger_usable = np.asarray(data["ledger_usable"], dtype=np.bool_)
    except (OSError, ValueError, KeyError) as error:
        raise Old308AnchorDatasetError("M2 join列を読めません") from error
    rows = CanonicalJoinRows(
        samples.state_ids, samples.source_groups, samples.game_keys,
        samples.labels, samples.weights, samples.folds, available_ms, segment,
        samples.ledger_values, samples.ledger_availability, ledger_usable,
    )
    _validate_canonical(rows, manifest)
    return rows, manifest


def _validate_canonical(rows: CanonicalJoinRows, manifest: Mapping[str, Any]) -> None:
    count = len(rows.state_ids)
    arrays = tuple(getattr(rows, field) for field in rows.__dataclass_fields__)
    if count != EXPECTED_STATE_COUNT or any(len(value) != count for value in arrays):
        raise Old308AnchorDatasetError(f"M2 canonical行数が不正です: {count}")
    if int(manifest.get("source_count", -1)) != EXPECTED_CANONICAL_SOURCE_COUNT:
        raise Old308AnchorDatasetError("M2 canonical元動画数が不正です")
    if len(set(map(str, rows.source_groups))) != EXPECTED_CANONICAL_SOURCE_COUNT:
        raise Old308AnchorDatasetError("M2 canonical source group数が不正です")
    if rows.ledger_values.shape[1:] != (2, 6):
        raise Old308AnchorDatasetError("6原始ledger valueではありません")
    if rows.ledger_availability.shape[1:] != (2, 6, 5):
        raise Old308AnchorDatasetError("6原始ledger availabilityではありません")


def _root_video_dirs(root: Path) -> list[Path]:
    summary_path = root / "SUMMARY.json"
    complete = _load_json(root / "COMPLETE")
    if complete.get("summary_sha256") != m2_builder.file_sha256(summary_path):
        raise Old308AnchorDatasetError(f"旧特徴量root receiptが不正です: {root}")
    summary = _load_json(summary_path)
    if summary.get("schema_version") != FEATURE_TABLE_VERSION:
        raise Old308AnchorDatasetError("旧特徴量root schemaが不正です")
    videos = summary.get("videos")
    if not isinstance(videos, list) or not videos:
        raise Old308AnchorDatasetError("旧特徴量動画一覧がありません")
    return [root / "schema=v1" / f"video=video_{row['target_id']}" for row in videos]


def _feature_source(video_dir: Path) -> LegacyFeatureSource:
    manifest_path = video_dir / "manifest.json"
    manifest_sha = m2_builder.file_sha256(manifest_path)
    if _load_json(video_dir / "COMPLETE").get("manifest_sha256") != manifest_sha:
        raise Old308AnchorDatasetError(f"旧特徴量video receiptが不正です: {video_dir}")
    manifest = _load_json(manifest_path)
    feature_path = video_dir / "features.parquet"
    feature = manifest.get("features", {})
    names = manifest.get("feature_names", {}).get("A", ())
    if manifest.get("schema_version") != FEATURE_TABLE_VERSION or len(names) != EXPECTED_FEATURE_COUNT:
        raise Old308AnchorDatasetError(f"旧A 308列schemaが不正です: {video_dir}")
    actual_sha = m2_builder.file_sha256(feature_path)
    if not isinstance(feature, Mapping) or feature.get("sha256") != actual_sha:
        raise Old308AnchorDatasetError(f"旧特徴量parquet hashが不正です: {video_dir}")
    return LegacyFeatureSource(
        str(manifest["target_id"]), str(manifest["source_group_id"]),
        int(manifest["partition_fold"]), feature_path, actual_sha,
        manifest_path, manifest_sha, tuple(map(str, names)), int(manifest["row_count"]),
    )


def _load_feature_sources(roots: Sequence[Path]) -> tuple[LegacyFeatureSource, ...]:
    sources = tuple(_feature_source(path) for root in roots for path in _root_video_dirs(root))
    groups = [source.source_group_id for source in sources]
    if len(sources) != EXPECTED_LEGACY_SOURCE_COUNT or len(set(groups)) != len(groups):
        raise Old308AnchorDatasetError("旧特徴量48動画が一意に揃っていません")
    first = sources[0].feature_names
    if any(source.feature_names != first for source in sources[1:]):
        raise Old308AnchorDatasetError("旧特徴量動画間でA列順が一致しません")
    return sources


def _join_key(group: object, available_ms: object, segment: object) -> tuple[str, int, int]:
    return str(group), int(available_ms), int(segment)


def _canonical_index(rows: CanonicalJoinRows) -> dict[tuple[str, int, int], int]:
    keys = [
        _join_key(*values) for values in zip(
            rows.source_groups, rows.available_ms, rows.online_segment_index, strict=True,
        )
    ]
    if len(set(keys)) != len(keys):
        raise Old308AnchorDatasetError("M2 canonical複合join keyが重複しています")
    return {key: index for index, key in enumerate(keys)}


def _read_feature_table(source: LegacyFeatureSource) -> Any:
    import pyarrow.parquet as parquet

    columns = [*JOIN_COLUMNS, *AUDIT_COLUMNS, *source.feature_names]
    try:
        table = parquet.read_table(source.feature_path, columns=columns)
    except (OSError, ValueError) as error:
        raise Old308AnchorDatasetError(f"旧特徴量parquetを読めません: {source.feature_path}") from error
    if table.num_rows != source.row_count:
        raise Old308AnchorDatasetError("旧特徴量parquet行数がmanifestと一致しません")
    return table


def _feature_values(table: Any, names: Sequence[str]) -> np.ndarray:
    frame = table.select(list(names)).to_pandas()
    try:
        return frame.to_numpy(dtype=np.float32, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise Old308AnchorDatasetError("旧308列をfloat32へ変換できません") from error


def _validate_match(
    meta: Mapping[str, Any], rows: CanonicalJoinRows, index: int,
) -> float | None:
    comparisons = (
        (str(meta["game_key"]), str(rows.game_keys[index]), "game_key"),
        (int(bool(meta["p1_won"])), int(rows.labels[index]), "p1_won"),
        (int(meta["partition_fold"]), int(rows.folds[index]), "partition_fold"),
    )
    for observed, expected, label in comparisons:
        if observed != expected:
            raise Old308AnchorDatasetError(f"複合join後の{label}が一致しません")
    value = meta["sample_weight"]
    if value is None:
        return None
    try:
        legacy_weight = float(value)
    except (TypeError, ValueError) as error:
        raise Old308AnchorDatasetError("旧sample_weightが数値ではありません") from error
    if not np.isfinite(legacy_weight):
        raise Old308AnchorDatasetError("旧sample_weightが有限ではありません")
    return abs(legacy_weight - float(rows.weights[index]))


def _join_source(
    source: LegacyFeatureSource, rows: CanonicalJoinRows,
    index: Mapping[tuple[str, int, int], int], features: np.ndarray,
    legacy_ids: np.ndarray, input_usable: np.ndarray, training_usable: np.ndarray,
    seen: np.ndarray,
) -> tuple[int, int, int, int, float]:
    table = _read_feature_table(source)
    metas = table.select([*JOIN_COLUMNS, *AUDIT_COLUMNS]).to_pylist()
    values = _feature_values(table, source.feature_names)
    local_keys: set[tuple[str, int, int]] = set()
    matched = equal_ids = weight_missing = weight_mismatches = 0
    max_weight_delta = 0.0
    for row_number, meta in enumerate(metas):
        key = _join_key(*(meta[name] for name in JOIN_COLUMNS))
        if key in local_keys:
            raise Old308AnchorDatasetError("旧特徴量複合join keyが動画内で重複しています")
        local_keys.add(key)
        position = index.get(key)
        if position is None:
            continue
        if seen[position]:
            raise Old308AnchorDatasetError("同じM2 canonical行へ複数joinしました")
        weight_delta = _validate_match(meta, rows, position)
        features[position], legacy_ids[position] = values[row_number], str(meta["state_id"])
        input_usable[position] = bool(meta["a_input_usable"])
        training_usable[position] = bool(meta["training_usable"])
        seen[position] = True
        matched += 1
        equal_ids += int(str(meta["state_id"]) == str(rows.state_ids[position]))
        weight_missing += int(weight_delta is None)
        if weight_delta is not None:
            weight_mismatches += int(not np.isclose(weight_delta, 0.0, atol=1e-8, rtol=1e-6))
            max_weight_delta = max(max_weight_delta, weight_delta)
    return matched, equal_ids, weight_missing, weight_mismatches, max_weight_delta


def _join_features(
    rows: CanonicalJoinRows, sources: Sequence[LegacyFeatureSource],
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    index = _canonical_index(rows)
    count, width = len(rows.state_ids), len(sources[0].feature_names)
    features = np.full((count, width), np.nan, dtype=np.float32)
    legacy_ids = np.full(count, "", dtype=f"<U{max(map(len, map(str, rows.state_ids)))}")
    input_usable = np.zeros(count, dtype=np.bool_)
    training_usable = np.zeros(count, dtype=np.bool_)
    seen = np.zeros(count, dtype=np.bool_)
    matched = equal_ids = weight_missing = weight_mismatches = 0
    max_weight_delta = 0.0
    selected = [source for source in sources if source.source_group_id in set(rows.source_groups)]
    if len(selected) != EXPECTED_CANONICAL_SOURCE_COUNT:
        raise Old308AnchorDatasetError("M2の46 source groupが旧特徴量に揃っていません")
    for source in selected:
        current = _join_source(
            source, rows, index, features, legacy_ids, input_usable, training_usable, seen,
        )
        matched, equal_ids = matched + current[0], equal_ids + current[1]
        weight_missing += current[2]
        weight_mismatches += current[3]
        max_weight_delta = max(max_weight_delta, current[4])
    if not bool(seen.all()):
        raise Old308AnchorDatasetError(f"旧308列のjoin漏れがあります: {int((~seen).sum())}行")
    return {
        "old308_features": features, "legacy_state_ids": legacy_ids,
        "anchor_input_usable": input_usable, "anchor_training_usable": training_usable,
    }, {
        "matched_state_count": matched, "same_state_id_count": equal_ids,
        "legacy_weight_missing_count": weight_missing,
        "legacy_weight_mismatch_count": weight_mismatches,
        "legacy_weight_max_absolute_delta": max_weight_delta,
    }


def _shortcut_columns(names: Sequence[str]) -> dict[str, list[str]]:
    return {
        group: [name for name in names if any(pattern in name for pattern in patterns)]
        for group, patterns in SHORTCUT_PATTERNS.items()
    }


def _source_receipts(sources: Sequence[LegacyFeatureSource]) -> list[dict[str, Any]]:
    return [{
        "target_id": source.target_id, "source_group_id": source.source_group_id,
        "partition_fold": source.partition_fold,
        "manifest_path": str(source.manifest_path.resolve()),
        "manifest_sha256": source.manifest_sha256,
        "features_path": str(source.feature_path.resolve()),
        "features_sha256": source.feature_sha256,
        "row_count": source.row_count,
    } for source in sources]


def _output_arrays(rows: CanonicalJoinRows, joined: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return dict(joined) | {
        name: np.asarray(getattr(rows, name)) for name in rows.__dataclass_fields__
    }


def _historical_anchor_manifest() -> dict[str, Any]:
    """再現対象の旧308列anchor契約を返す。"""

    return {
        "name": "A_common_original", "model_family": "tree",
        "estimator": "HistGradientBoostingClassifier",
        "model_params": dict(MODEL_PARAMS),
        "side_swap": "mirror_augmentation_and_symmetric_average",
        "fixed_fold": "eval_k_tune_k_plus_1_train_remaining_4",
        "calibration": "tune_fold_symmetric_zero_intercept_platt",
    }


def _manifest(
    args: argparse.Namespace, rows: CanonicalJoinRows, names: Sequence[str],
    sources: Sequence[LegacyFeatureSource], arrays: Mapping[str, np.ndarray],
    audit: Mapping[str, int | float], dataset_path: Path,
) -> dict[str, Any]:
    shortcuts = _shortcut_columns(names)
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "role": "diagnostic_baseline_only_not_next_mainline_foundation",
        "state_count": len(rows.state_ids), "source_count": len(set(rows.source_groups)),
        "game_count": len(set(rows.game_keys)), "feature_count": len(names),
        "feature_names": list(names),
        "feature_names_logical_sha256": m2_builder._array_sha256(np.asarray(names)),
        "feature_schema_version": FEATURE_TABLE_VERSION,
        "join_contract": {
            "primary_key": list(JOIN_COLUMNS), "state_id_is_primary_key": False,
            "cross_checks": ["game_key", "p1_won", "partition_fold"],
            **dict(audit),
        },
        "weight_contract": {
            "source": "m2_canonical_dataset",
            "legacy_sample_weight_used": False,
            "policy": "same_fixed_fold_game_equal_weight_as_m2",
        },
        "shortcut_columns_included_for_diagnostic_reproduction": shortcuts,
        "shortcut_column_count": sum(map(len, shortcuts.values())),
        "historical_anchor": _historical_anchor_manifest(),
        "ledger": {
            "field_count_per_side": 6, "typed_values_and_availability": True,
            "role": "zero_initialized_residual_candidate",
        },
        "attack_difference_ten_percent_correction": False,
        "uses_337_or_338_feature_family": False,
        "production_config_changed": False,
        "m2_dataset_root": str(args.m2_root.resolve()),
        "m2_manifest_sha256": m2_builder.file_sha256(args.m2_root / "manifest.json"),
        "m2_dataset_sha256": m2_builder.file_sha256(args.m2_root / "dataset.npz"),
        "legacy_feature_roots": [str(path.resolve()) for path in args.feature_roots],
        "selected_legacy_sources": _source_receipts([
            source for source in sources if source.source_group_id in set(rows.source_groups)
        ]),
        "array_sha256": {name: m2_builder._array_sha256(value) for name, value in arrays.items()},
        "dataset": {"name": dataset_path.name, "sha256": m2_builder.file_sha256(dataset_path)},
        "code_sha256": {"builder": m2_builder.file_sha256(Path(__file__))},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """旧308列をM2 canonical時点へ結合し、排他的に保存する。"""

    if args.output_root.exists():
        raise Old308AnchorDatasetError(f"出力先は新規必須です: {args.output_root}")
    rows, _ = _load_canonical(args.m2_root)
    sources = _load_feature_sources(args.feature_roots)
    joined, audit = _join_features(rows, sources)
    arrays = _output_arrays(rows, joined)
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    m2_builder._write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(
        args, rows, sources[0].feature_names, sources, arrays, audit, dataset_path,
    )
    manifest_path = args.output_root / "manifest.json"
    m2_builder._write_json_exclusive(manifest_path, manifest)
    m2_builder._write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "manifest_sha256": m2_builder.file_sha256(manifest_path),
        "dataset_sha256": m2_builder.file_sha256(dataset_path),
    })
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=DEFAULT_M2_ROOT)
    parser.add_argument("--feature-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.feature_roots = tuple(args.feature_root or DEFAULT_FEATURE_ROOTS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({
        key: manifest[key] for key in ("state_count", "source_count", "feature_count")
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
