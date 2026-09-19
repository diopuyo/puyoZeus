"""固定M1 V2 datasetへ全消し状態と材料・空間8補助教師を追加する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import build_advantage_m1_dataset_v2 as m1_builder
from scripts import train_advantage_m1_zero_counterfactual_v3 as m1_trainer
from src import advantage_m2_auxiliary_cnn_v1 as m2
from src.projected_state_tensorizer_v1 import (
    AUXILIARY_FEATURE_NAMES,
    AUXILIARY_MANIFEST_PATH,
    AUXILIARY_MANIFEST_SHA256,
)


FORMAT_VERSION = "advantage-m2-auxiliary-dataset/v1"
COMPLETE_VERSION = "advantage-m2-auxiliary-dataset-complete/v1"
DEFAULT_PARENT_ROOT = Path(
    "data/verify/advantage_m1_canonical_dataset_46v_2026-09-05_v3_finalized_primary6"
)
PRODUCTION_CONFIG_PATH = Path("src/production_config.py")
ALL_CLEAR_COLUMNS = (
    "state_id", "a_p1_all_clear_pending", "a_p2_all_clear_pending",
)
SHORTCUTS_EXCLUDED = (
    "score", "tsumo_count", "player_identity", "recognition_quality",
    "mechanism_label", "cumulative_match_score", "operation_speed",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM2DatasetError(RuntimeError):
    """M2 datasetの結合、凍結資産、排他保存契約に違反した。"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_arrays(samples: m1_trainer.CanonicalSamplesV3) -> dict[str, np.ndarray]:
    return {
        name: getattr(samples, name) for name in (
            "boards", "queues", "ledger_values", "ledger_availability", "labels",
            "source_groups", "game_keys", "weights", "state_ids", "available_ms",
            "online_segment_index", "ledger_usable", "projected_input_digest",
            "projected_gate_status",
        )
    }


def _load_parent(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        samples, manifest = m1_trainer.load_canonical_dataset_v2(root)
    except (OSError, ValueError, RuntimeError) as error:
        raise AdvantageM2DatasetError("親M1 datasetの検証に失敗しました") from error
    arrays = _parent_arrays(samples)
    if len(arrays["state_ids"]) != len(set(map(str, arrays["state_ids"]))):
        raise AdvantageM2DatasetError("親M1 state_idが重複しています")
    return arrays, manifest


def _read_all_clear_rows(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    try:
        return parquet.read_table(
            root / "states.parquet", columns=list(ALL_CLEAR_COLUMNS),
        ).to_pylist()
    except (OSError, ValueError) as error:
        raise AdvantageM2DatasetError(f"全消し列を読めません: {root}") from error


def _validate_source(source: Mapping[str, Any]) -> Path:
    root = Path(str(source.get("table_root", "")))
    manifest_path = root / "manifest.json"
    expected = str(source.get("table_manifest_sha256", ""))
    if not manifest_path.is_file() or file_sha256(manifest_path) != expected:
        raise AdvantageM2DatasetError(f"親M1 source manifestが変化しています: {root}")
    return root


def _all_clear_arrays(
    state_ids: np.ndarray, sources: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    index = {str(state_id): row for row, state_id in enumerate(state_ids)}
    values = np.zeros((len(state_ids), 2), dtype=np.float32)
    masks = np.zeros((len(state_ids), 2, m2.AVAILABILITY_COUNT), dtype=np.float32)
    seen = np.zeros(len(state_ids), dtype=np.bool_)
    audits: list[dict[str, Any]] = []
    for source in sources:
        root = _validate_source(source)
        matched = _join_source_all_clear(index, _read_all_clear_rows(root), values, masks, seen)
        audits.append({
            "target_id": str(source["target_id"]), "table_root": str(root.resolve()),
            "table_manifest_sha256": str(source["table_manifest_sha256"]),
            "matched_state_count": matched,
        })
    if not bool(seen.all()):
        raise AdvantageM2DatasetError(f"全消しjoin漏れがあります: {int((~seen).sum())}行")
    return values, masks, audits


def _join_source_all_clear(
    index: Mapping[str, int], rows: Sequence[Mapping[str, Any]],
    values: np.ndarray, masks: np.ndarray, seen: np.ndarray,
) -> int:
    matched = 0
    for state in rows:
        position = index.get(str(state.get("state_id", "")))
        if position is None:
            continue
        if seen[position]:
            raise AdvantageM2DatasetError("同じM1 state_idへ複数行がjoinしました")
        row_values, row_masks = m2.tensorize_materialized_all_clear_v1(state)
        values[position], masks[position], seen[position] = row_values, row_masks, True
        matched += 1
    return matched


def _auxiliary_arrays(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    shape = (len(boards), 2, len(AUXILIARY_FEATURE_NAMES))
    targets = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=np.bool_)
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for row, pair in enumerate(boards):
        for side, board in enumerate(pair):
            key = np.ascontiguousarray(board).tobytes()
            result = cache.get(key)
            if result is None:
                result = m2.auxiliary_target_from_category_board_v1(board)
                cache[key] = result
            targets[row, side], masks[row, side] = result
        if (row + 1) % 50_000 == 0:
            print(f"auxiliary targets: {row + 1}/{len(boards)}", flush=True)
    return targets, masks, len(cache)


def _array_sha256(value: np.ndarray) -> str:
    return m1_builder._array_sha256(value)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _availability_counts(masks: np.ndarray) -> dict[str, int]:
    indices = masks.argmax(axis=-1).reshape(-1)
    return {
        state.value: int(np.count_nonzero(indices == position))
        for position, state in enumerate(m2.AVAILABILITY_ORDER)
    }


def _receipt(
    arrays: Mapping[str, np.ndarray], parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent_ids = parent_manifest["shared_tensorizer_full_receipt"][
        "state_id_ordered_sha256"
    ]
    current_ids = _array_sha256(arrays["state_ids"])
    if current_ids != parent_ids:
        raise AdvantageM2DatasetError("親M1とstate_id順が一致しません")
    names = (
        "all_clear_values", "all_clear_availability",
        "auxiliary_targets", "auxiliary_mask",
    )
    return {
        "row_count": len(arrays["state_ids"]), "state_id_ordered_sha256": current_ids,
        "parent_state_id_ordered_sha256": parent_ids,
        "array_sha256": {name: _array_sha256(arrays[name]) for name in names},
        "m2_input_schema_version": m2.M2_INPUT_SCHEMA_VERSION,
        "auxiliary_feature_order": list(AUXILIARY_FEATURE_NAMES),
    }


def _manifest(
    args: argparse.Namespace, arrays: Mapping[str, np.ndarray],
    parent_manifest: Mapping[str, Any], audits: Sequence[Mapping[str, Any]],
    unique_boards: int, dataset_path: Path,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "source_count": int(parent_manifest["source_count"]),
        "game_count": int(parent_manifest["game_count"]),
        "state_count": len(arrays["state_ids"]),
        "parent_dataset_root": str(args.parent_root.resolve()),
        "parent_manifest_sha256": file_sha256(args.parent_root / "manifest.json"),
        "parent_complete_sha256": file_sha256(args.parent_root / "COMPLETE"),
        "parent_dataset_sha256": file_sha256(args.parent_root / "dataset.npz"),
        "same_parent_rows_and_folds": True,
        "all_clear_availability_counts": _availability_counts(
            arrays["all_clear_availability"],
        ),
        "auxiliary_manifest_path": str(AUXILIARY_MANIFEST_PATH.resolve()),
        "auxiliary_manifest_sha256": AUXILIARY_MANIFEST_SHA256,
        "auxiliary_unique_board_count": unique_boards,
        "auxiliary_role": "training_only_side_head_not_win_probability_input",
        "shortcuts_excluded": list(SHORTCUTS_EXCLUDED),
        "m2_receipt": _receipt(arrays, parent_manifest),
        "sources": list(audits),
        "code_sha256": {
            "builder": file_sha256(Path(__file__)),
            "m2_core": file_sha256(REPO_ROOT / "src/advantage_m2_auxiliary_cnn_v1.py"),
            "auxiliary_tensorizer": file_sha256(
                REPO_ROOT / "src/projected_state_tensorizer_v1.py"
            ),
            "production_config": file_sha256(REPO_ROOT / PRODUCTION_CONFIG_PATH),
        },
        "dataset": {"name": dataset_path.name, "sha256": file_sha256(dataset_path)},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """親M1の行とfoldを固定したままM2追加列を排他的に保存する。"""

    if args.output_root.exists():
        raise AdvantageM2DatasetError(f"出力先は新規必須です: {args.output_root}")
    arrays, parent_manifest = _load_parent(args.parent_root)
    all_clear = _all_clear_arrays(arrays["state_ids"], parent_manifest["sources"])
    targets, target_masks, unique_boards = _auxiliary_arrays(arrays["boards"])
    arrays |= {
        "all_clear_values": all_clear[0], "all_clear_availability": all_clear[1],
        "auxiliary_targets": targets, "auxiliary_mask": target_masks,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    _write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(
        args, arrays, parent_manifest, all_clear[2], unique_boards, dataset_path,
    )
    manifest_path = args.output_root / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": file_sha256(dataset_path),
    })
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({
        key: manifest[key] for key in (
            "source_count", "game_count", "state_count",
            "all_clear_availability_counts", "auxiliary_unique_board_count",
        )
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
