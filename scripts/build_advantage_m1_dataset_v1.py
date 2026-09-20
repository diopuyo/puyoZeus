"""監査済み46動画のevent stateからM0/M1共通学習集合を固定する。"""

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

from scripts import train_advantage_m0_current_cnn_v1 as m0
from src.advantage_m1_causal_ledger_v1 import (
    AdvantageM1InputsV1,
    advantage_m1_inputs_from_materialized_state,
    materialized_ledger_is_usable,
)


FORMAT_VERSION = "advantage-m1-canonical-dataset/v1"
COMPLETE_VERSION = "advantage-m1-canonical-dataset-complete/v1"
DEFAULT_INDEX_ROOT = Path("data/verify/board_training_source_index_dev97_2026-09-04_v5")
DEFAULT_TABLE_ROOTS = (
    Path("data/verify/attack_quality_causal_v2_rebuild_2026-09-01/learning_first30"),
    Path("data/verify/attack_quality_causal_v2_rebuild_2026-09-01/learning_remaining18"),
)
EXPECTED_SOURCE_COUNT = 46
MAX_STATES_PER_GAME = 96
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM1DatasetError(RuntimeError):
    """M1学習集合の固定契約に違反した。"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdvantageM1DatasetError(f"JSON objectではありません: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _pilot_sources(index_root: Path) -> tuple[dict[str, Any], ...]:
    summary = m0.load_index(index_root)
    sources = tuple(
        dict(row) for row in summary["sources"] if row.get("dataset_role") == "pilot"
    )
    if len(sources) != EXPECTED_SOURCE_COUNT:
        raise AdvantageM1DatasetError(
            f"品質通過pilotが{EXPECTED_SOURCE_COUNT}本ではありません: {len(sources)}"
        )
    groups = [str(row["source_group_id"]) for row in sources]
    if len(groups) != len(set(groups)):
        raise AdvantageM1DatasetError("pilot source groupが重複しています")
    return sources


def _table_dir(target_id: str, roots: Sequence[Path]) -> Path:
    candidates = [
        root / "schema=v1" / f"video=video_{target_id}" for root in roots
        if (root / "schema=v1" / f"video=video_{target_id}").is_dir()
    ]
    if len(candidates) != 1:
        raise AdvantageM1DatasetError(
            f"{target_id}の学習表が一意ではありません: {candidates}"
        )
    return candidates[0]


def _validate_table_root(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    manifest, complete = _load_json(manifest_path), _load_json(complete_path)
    if complete.get("manifest_sha256") != file_sha256(manifest_path):
        raise AdvantageM1DatasetError(f"学習表の完了hashが不一致です: {root}")
    expected = (f"video_{source['target_id']}", str(source["source_group_id"]))
    actual = (manifest.get("source_video_id"), manifest.get("source_group_id"))
    if actual != expected:
        raise AdvantageM1DatasetError(f"学習表のsource identityが不一致です: {root}")
    for name in ("states", "labels"):
        entry = manifest.get("tables", {}).get(name, {})
        # manifest名は固定表名と厳密一致のみ許可（別名・別dir・絶対path・親参照を救済しない）
        if entry.get("name") != f"{name}.parquet":
            raise AdvantageM1DatasetError(f"{name} table名が固定名ではありません: {root}")
        path = root / f"{name}.parquet"
        if not path.is_file() or entry.get("sha256") != file_sha256(path):
            raise AdvantageM1DatasetError(f"{name} table hashが不一致です: {root}")
    return manifest


def _read_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as parquet

    states = parquet.read_table(root / "states.parquet").to_pylist()
    labels = parquet.read_table(root / "labels.parquet").to_pylist()
    if len(states) != len(labels):
        raise AdvantageM1DatasetError(f"state/label行数が不一致です: {root}")
    return states, labels


def _usable_records(
    states: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]],
) -> list[tuple[AdvantageM1InputsV1, float, str, int, int, bool]]:
    label_by_state = {str(row["state_id"]): row for row in labels}
    records = []
    for state in states:
        label = label_by_state.get(str(state["state_id"]))
        if label is None or not _row_is_usable(state, label):
            continue
        inputs = advantage_m1_inputs_from_materialized_state(state)
        records.append((
            inputs, float(bool(label["p1_won"])), str(label["game_key"]),
            int(state["available_ms"]), int(state["online_segment_index"]),
            materialized_ledger_is_usable(state),
        ))
    return _limit_records(records)


def _row_is_usable(state: Mapping[str, Any], label: Mapping[str, Any]) -> bool:
    return bool(
        state.get("a_input_usable") is True
        and label.get("win_label_available") is True
        and isinstance(label.get("p1_won"), bool)
        and isinstance(label.get("game_key"), str)
    )


def _limit_records(
    records: Sequence[tuple[AdvantageM1InputsV1, float, str, int, int, bool]],
) -> list[tuple[AdvantageM1InputsV1, float, str, int, int, bool]]:
    grouped: dict[str, list[tuple[AdvantageM1InputsV1, float, str, int, int, bool]]] = {}
    for row in records:
        grouped.setdefault(row[2], []).append(row)
    selected = []
    for rows in grouped.values():
        if len(rows) <= MAX_STATES_PER_GAME:
            selected.extend(rows)
            continue
        positions = np.linspace(0, len(rows) - 1, MAX_STATES_PER_GAME, dtype=np.int64)
        selected.extend(rows[int(position)] for position in positions)
    return selected


def _source_arrays(
    source: Mapping[str, Any], roots: Sequence[Path],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = _table_dir(str(source["target_id"]), roots)
    manifest = _validate_table_root(root, source)
    states, labels = _read_rows(root)
    records = _usable_records(states, labels)
    if not records:
        raise AdvantageM1DatasetError(f"有効な学習行がありません: {source['target_id']}")
    arrays = _records_to_arrays(records, str(source["source_group_id"]))
    audit = {
        "target_id": str(source["target_id"]), "table_root": str(root.resolve()),
        "table_manifest_sha256": file_sha256(root / "manifest.json"),
        "selected_state_count": len(records),
        "game_count": len(np.unique(arrays["game_keys"])),
        "ledger_usable_count": int(arrays["ledger_usable"].sum()),
        "partition_fold": int(manifest["partition_fold"]), "tier": str(source["tier"]),
    }
    return arrays, audit


def _records_to_arrays(
    records: Sequence[tuple[AdvantageM1InputsV1, float, str, int, int, bool]],
    source_group: str,
) -> dict[str, np.ndarray]:
    return {
        "boards": np.stack([row[0].boards for row in records]).astype(np.int8),
        "queues": np.stack([row[0].queues for row in records]).astype(np.int8),
        "ledger_values": np.stack([row[0].ledger_values for row in records]),
        "ledger_availability": np.stack([row[0].ledger_availability for row in records]),
        "labels": np.asarray([row[1] for row in records], dtype=np.float32),
        "game_keys": np.asarray([row[2] for row in records]),
        "available_ms": np.asarray([row[3] for row in records], dtype=np.int64),
        "online_segment_index": np.asarray([row[4] for row in records], dtype=np.int32),
        "ledger_usable": np.asarray([row[5] for row in records], dtype=np.bool_),
        "source_groups": np.full(len(records), source_group),
    }


def _concatenate(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = tuple(parts[0])
    if any(tuple(part) != names for part in parts):
        raise AdvantageM1DatasetError("source間でarray schemaが一致しません")
    output = {name: np.concatenate([part[name] for part in parts]) for name in names}
    output["weights"] = m0._equal_game_weights(output["game_keys"])
    return output


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict[str, Any]:
    """監査済み表を読み、M0/M1が共有する固定行集合を新規保存する。"""

    if args.output_root.exists():
        raise AdvantageM1DatasetError(f"出力先は新規必須です: {args.output_root}")
    sources = _pilot_sources(args.index_root)
    pairs = [_source_arrays(source, args.table_roots) for source in sources]
    arrays = _concatenate([pair[0] for pair in pairs])
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    _write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(args, sources, arrays, [pair[1] for pair in pairs], dataset_path)
    manifest_path = args.output_root / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": COMPLETE_VERSION,
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": file_sha256(dataset_path),
    })
    return manifest


def _manifest(
    args: argparse.Namespace, sources: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray], audits: Sequence[Mapping[str, Any]],
    dataset_path: Path,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "source_count": len(sources), "state_count": len(arrays["labels"]),
        "game_count": len(np.unique(arrays["game_keys"])),
        "ledger_usable_count": int(arrays["ledger_usable"].sum()),
        "max_states_per_game": MAX_STATES_PER_GAME,
        "equal_total_weight_per_game": True,
        "source_index_root": str(args.index_root.resolve()),
        "source_index_sha256": file_sha256(args.index_root / "SUMMARY.json"),
        "target_ids": [str(row["target_id"]) for row in sources],
        "source_group_ids": [str(row["source_group_id"]) for row in sources],
        "tier_counts": dict(sorted(Counter(str(row["tier"]) for row in sources).items())),
        "training_row_policy": "a_input_usable_and_official_win_label_available",
        "ledger_policy": "raw_value_plus_five_state_availability_mask",
        "canonical_materialized_parity": True,
        "sources": list(audits),
        "code_sha256": {
            "builder": file_sha256(Path(__file__)),
            "tensorizer": file_sha256(REPO_ROOT / "src/advantage_m1_causal_ledger_v1.py"),
        },
        "dataset": {"name": dataset_path.name, "sha256": file_sha256(dataset_path)},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--table-roots", type=Path, nargs="+", default=list(DEFAULT_TABLE_ROOTS))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(json.dumps({
        key: manifest[key] for key in (
            "source_count", "game_count", "state_count", "ledger_usable_count",
        )
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
