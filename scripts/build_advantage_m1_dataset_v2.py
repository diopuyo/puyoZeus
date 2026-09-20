"""監査済み46動画から確定会計分離済みM1 V2学習集合を固定する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts import train_advantage_m0_current_cnn_v1 as m0
from scripts.build_event_learning_tables_v1 import VIDEO_COMPLETE_VERSION
from src import advantage_m1_causal_ledger_v3 as ledger
from src.canonical_observation_adapter_v3 import CANONICAL_ADAPTER_VERSION
from src.canonical_observation_v3 import CANONICAL_OBSERVATION_SCHEMA_VERSION
from src.event_learning_tables_v1 import LEARNING_TABLE_VERSION
from src.projected_state_dataset_v1 import (
    iter_projected_state_dataset,
    validate_projected_state_dataset,
)


FORMAT_VERSION = "advantage-m1-canonical-dataset/v2"
COMPLETE_VERSION = "advantage-m1-canonical-dataset-complete/v2"
PROJECTED_NOT_JOINED = "not_joined"
DEFAULT_INDEX_ROOT = Path("data/verify/board_training_source_index_dev97_2026-09-04_v5")
DEFAULT_TABLE_ROOTS = (
    Path("data/verify/attack_quality_causal_v6_finalized_pending_guard_2026-09-04/learning_first30"),
    Path("data/verify/attack_quality_causal_v6_finalized_pending_guard_2026-09-04/learning_remaining18"),
)
DEFAULT_PROJECTED_ROOT = Path("data/verify/projected_state_dataset_pilot48_2026-09-04_v1")
EXPECTED_SOURCE_COUNT = 46
MAX_STATES_PER_GAME = 96
REPO_ROOT = Path(__file__).resolve().parents[1]


class AdvantageM1DatasetV2Error(RuntimeError):
    """M1 V2学習集合または入力資産の固定契約に違反した。"""


@dataclass(frozen=True, slots=True)
class ProjectedJoinIndex:
    """検証済みprojected観測のjoin索引と固定receipt。"""

    by_state_id: Mapping[str, tuple[str, str]]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingRecordV2:
    """一つの選択済み学習行。"""

    inputs: ledger.AdvantageM1InputsV3
    label: float
    game_key: str
    available_ms: int
    online_segment_index: int
    ledger_usable: bool
    state_id: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM1DatasetV2Error(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM1DatasetV2Error(f"JSON objectではありません: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _pilot_sources(
    index_root: Path, increment_manifest: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    summary = m0.load_index(index_root)
    base_sources = tuple(
        dict(row) for row in summary["sources"] if row.get("dataset_role") == "pilot"
    )
    if len(base_sources) != EXPECTED_SOURCE_COUNT:
        raise AdvantageM1DatasetV2Error(
            f"品質通過pilotが{EXPECTED_SOURCE_COUNT}本ではありません: {len(base_sources)}"
        )
    increments = _increment_sources(summary, increment_manifest)
    sources = base_sources + increments
    groups = [str(row["source_group_id"]) for row in sources]
    if len(groups) != len(set(groups)):
        raise AdvantageM1DatasetV2Error("pilot source groupが重複しています")
    return sources


def _increment_sources(
    summary: Mapping[str, Any], manifest_path: Path | None,
) -> tuple[dict[str, Any], ...]:
    if manifest_path is None:
        return ()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    existing = {str(row["target_id"]): row for row in summary["sources"]}
    result: list[dict[str, Any]] = []
    for row in rows:
        target = str(row.get("target_id") or "")
        source = dict(existing.get(target, {}))
        checks = (
            row.get("format_version") == "advantage-m2-increment-source/v1",
            row.get("role") == "development_increment_only",
            source.get("dataset_role") == "historical_development",
            source.get("source_group_id") == row.get("youtube_id"),
            source.get("tier") == row.get("tier"),
            source.get("npz_sha256") == row.get("board_npz_sha256"),
            int(source.get("row_count", -1)) == int(row.get("row_count") or -2),
            int(row.get("fold") or 0) in range(1, 7),
        )
        if not target or not all(checks):
            raise AdvantageM1DatasetV2Error(f"増分source契約が不正です: {target}")
        source.update({
            "dataset_role": "development_increment",
            "partition_fold": int(row["fold"]),
            "canonical_video_alias": str(row["canonical_video_alias"]),
        })
        result.append(source)
    if not result or len({row["target_id"] for row in result}) != len(result):
        raise AdvantageM1DatasetV2Error("増分sourceが空または重複しています")
    return tuple(result)


def _table_dir(source: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    target_id = str(source["target_id"])
    source_video_id = str(
        source.get("canonical_video_alias") or f"video_{target_id}"
    )
    candidates = [
        root / "schema=v1" / f"video={source_video_id}" for root in roots
        if (root / "schema=v1" / f"video={source_video_id}").is_dir()
    ]
    if len(candidates) != 1:
        raise AdvantageM1DatasetV2Error(
            f"{target_id}のV6学習表が一意ではありません: {candidates}"
        )
    return candidates[0]


def _validate_table_root(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    manifest, complete = _load_json(manifest_path), _load_json(complete_path)
    if complete.get("manifest_sha256") != file_sha256(manifest_path):
        raise AdvantageM1DatasetV2Error(f"学習表の完了hashが不一致です: {root}")
    if complete.get("format_version") != VIDEO_COMPLETE_VERSION:
        raise AdvantageM1DatasetV2Error(f"学習表のCOMPLETE形式が不正です: {root}")
    if manifest.get("schema_version") != LEARNING_TABLE_VERSION:
        raise AdvantageM1DatasetV2Error(f"学習表のschema versionが不正です: {root}")
    expected = _expected_source_identity(source)
    actual = _manifest_source_identity(manifest)
    if actual != expected:
        raise AdvantageM1DatasetV2Error(f"学習表のsource identityが不一致です: {root}")
    for name in ("states", "labels"):
        entry = manifest.get("tables", {}).get(name, {})
        # manifest名は固定表名と厳密一致のみ許可（別名・別dir・絶対path・親参照を救済しない）
        if entry.get("name") != f"{name}.parquet":
            raise AdvantageM1DatasetV2Error(
                f"{name} table名が固定名ではありません: {root}"
            )
        path = root / f"{name}.parquet"
        if not path.is_file() or entry.get("sha256") != file_sha256(path):
            raise AdvantageM1DatasetV2Error(f"{name} table hashが不一致です: {root}")
    return manifest


def _expected_source_identity(source: Mapping[str, Any]) -> tuple[Any, ...]:
    target = str(source["target_id"])
    return (
        str(source.get("canonical_video_alias") or f"video_{target}"),
        str(source["source_group_id"]), target,
        str(source["tier"]), int(source["partition_fold"]),
    )


def _manifest_source_identity(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        manifest.get("source_video_id"), manifest.get("source_group_id"),
        str(manifest.get("target_id")), str(manifest.get("tier")),
        int(manifest.get("partition_fold", -1)),
    )


def _read_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as parquet

    states = parquet.read_table(root / "states.parquet").to_pylist()
    labels = parquet.read_table(root / "labels.parquet").to_pylist()
    if len(states) != len(labels):
        raise AdvantageM1DatasetV2Error(f"state/label行数が不一致です: {root}")
    return states, labels


def _label_index(labels: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in labels:
        state_id = str(row.get("state_id", ""))
        if not state_id or state_id in output:
            raise AdvantageM1DatasetV2Error("label state_idが空または重複しています")
        output[state_id] = row
    return output


def _usable_records(
    states: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]],
) -> list[TrainingRecordV2]:
    label_by_state = _label_index(labels)
    records: list[TrainingRecordV2] = []
    seen: set[str] = set()
    for state in states:
        state_id = str(state.get("state_id", ""))
        if not state_id or state_id in seen:
            raise AdvantageM1DatasetV2Error("state state_idが空または重複しています")
        seen.add(state_id)
        label = label_by_state.get(state_id)
        if label is None or not _row_is_usable(state, label):
            continue
        records.append(_record(state, label, state_id))
    if seen != set(label_by_state):
        raise AdvantageM1DatasetV2Error("state/labelのstate_id集合が一致しません")
    return _limit_records(records)


def _record(
    state: Mapping[str, Any], label: Mapping[str, Any], state_id: str,
) -> TrainingRecordV2:
    primary = ledger.advantage_m1_primary_from_materialized_state(state)
    return TrainingRecordV2(
        inputs=ledger.tensorize_primary(primary),
        label=float(bool(label["p1_won"])), game_key=str(label["game_key"]),
        available_ms=int(state["available_ms"]),
        online_segment_index=int(state["online_segment_index"]),
        ledger_usable=ledger.materialized_ledger_is_usable(state), state_id=state_id,
    )


def _row_is_usable(state: Mapping[str, Any], label: Mapping[str, Any]) -> bool:
    return bool(
        state.get("a_input_usable") is True
        and label.get("win_label_available") is True
        and isinstance(label.get("p1_won"), bool)
        and isinstance(label.get("game_key"), str)
    )


def _limit_records(records: Sequence[TrainingRecordV2]) -> list[TrainingRecordV2]:
    grouped: dict[str, list[TrainingRecordV2]] = {}
    for row in records:
        grouped.setdefault(row.game_key, []).append(row)
    selected: list[TrainingRecordV2] = []
    for rows in grouped.values():
        if len(rows) <= MAX_STATES_PER_GAME:
            selected.extend(rows)
            continue
        positions = np.linspace(0, len(rows) - 1, MAX_STATES_PER_GAME, dtype=np.int64)
        selected.extend(rows[int(position)] for position in positions)
    return selected


def _projected_index(root: Path) -> ProjectedJoinIndex:
    receipt = validate_projected_state_dataset(root)
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    manifest, complete = _load_json(manifest_path), _load_json(complete_path)
    records = _projected_records(root, manifest)
    if len(records) != receipt.record_count:
        raise AdvantageM1DatasetV2Error("projected validator件数とjoin索引件数が不一致です")
    fixed = {
        "root": str(root.resolve()), "record_count": receipt.record_count,
        "source_count": receipt.source_count,
        "manifest_sha256": receipt.manifest_sha256,
        "complete_sha256": file_sha256(complete_path),
        "observations_sha256": receipt.observations_sha256,
        "unique_index_sha256": str(manifest["unique_index_sha256"]),
        "schema_version": str(manifest["schema_version"]),
        "complete_format": str(complete["format"]),
        "input_digest_semantics": "ProjectedStateObservationV1.input_digest",
        "validator": "src.projected_state_dataset_v1.validate_projected_state_dataset",
    }
    return ProjectedJoinIndex(records, fixed)


def _projected_records(
    root: Path, manifest: Mapping[str, Any],
) -> dict[str, tuple[str, str]]:
    observations = iter_projected_state_dataset(root, _manifest=manifest)
    return _projected_records_from_observations(observations)


def _projected_records_from_observations(
    observations: Iterable[Any],
) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for value in observations:
        state_id, gate = str(value.observation_id), str(value.gate_status)
        if not state_id or state_id in records:
            raise AdvantageM1DatasetV2Error("projected observation_idが空または重複です")
        records[state_id] = (str(value.input_digest), gate)
    return records


def _source_arrays(
    source: Mapping[str, Any], roots: Sequence[Path], projected: ProjectedJoinIndex,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = _table_dir(source, roots)
    manifest = _validate_table_root(root, source)
    states, labels = _read_rows(root)
    records = _usable_records(states, labels)
    if not records:
        raise AdvantageM1DatasetV2Error(f"有効な学習行がありません: {source['target_id']}")
    arrays = _records_to_arrays(records, str(source["source_group_id"]), projected)
    audit = {
        "target_id": str(source["target_id"]), "table_root": str(root.resolve()),
        "table_manifest_sha256": file_sha256(root / "manifest.json"),
        "selected_state_count": len(records),
        "game_count": len(np.unique(arrays["game_keys"])),
        "ledger_usable_count": int(arrays["ledger_usable"].sum()),
        "projected_joined_count": int(np.count_nonzero(
            arrays["projected_gate_status"] != PROJECTED_NOT_JOINED
        )),
        "partition_fold": int(manifest["partition_fold"]), "tier": str(source["tier"]),
    }
    return arrays, audit


def _records_to_arrays(
    records: Sequence[TrainingRecordV2], source_group: str,
    projected: ProjectedJoinIndex,
) -> dict[str, np.ndarray]:
    joined = [projected.by_state_id.get(row.state_id) for row in records]
    return {
        "boards": np.stack([row.inputs.boards for row in records]).astype(np.int8),
        "queues": np.stack([row.inputs.queues for row in records]).astype(np.int8),
        "ledger_values": np.stack([row.inputs.ledger_values for row in records]),
        "ledger_availability": np.stack([
            row.inputs.ledger_availability for row in records
        ]),
        "labels": np.asarray([row.label for row in records], dtype=np.float32),
        "game_keys": np.asarray([row.game_key for row in records]),
        "available_ms": np.asarray([row.available_ms for row in records], dtype=np.int64),
        "online_segment_index": np.asarray([
            row.online_segment_index for row in records
        ], dtype=np.int32),
        "ledger_usable": np.asarray([row.ledger_usable for row in records], dtype=np.bool_),
        "source_groups": np.full(len(records), source_group),
        "state_ids": np.asarray([row.state_id for row in records]),
        "projected_input_digest": np.asarray([
            "" if item is None else item[0] for item in joined
        ]),
        "projected_gate_status": np.asarray([
            PROJECTED_NOT_JOINED if item is None else item[1] for item in joined
        ]),
    }


def _concatenate(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = tuple(parts[0])
    if any(tuple(part) != names for part in parts):
        raise AdvantageM1DatasetV2Error("source間でarray schemaが一致しません")
    output = {name: np.concatenate([part[name] for part in parts]) for name in names}
    state_ids = tuple(str(value) for value in output["state_ids"])
    if len(state_ids) != len(set(state_ids)):
        raise AdvantageM1DatasetV2Error("選択済みstate_idがsource間で重複しています")
    output["weights"] = m0._equal_game_weights(output["game_keys"])
    return output


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _shared_tensorizer_receipt(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values_sha = _array_sha256(arrays["ledger_values"])
    availability_sha = _array_sha256(arrays["ledger_availability"])
    combined = hashlib.sha256(
        f"{values_sha}\0{availability_sha}".encode("ascii")
    ).hexdigest()
    return {
        "row_count": len(arrays["state_ids"]), "failure_count": 0,
        "state_id_ordered_sha256": _array_sha256(arrays["state_ids"]),
        "ledger_values_sha256": values_sha,
        "ledger_availability_sha256": availability_sha,
        "ledger_tensor_sha256": combined,
        "ledger_tensor_sha256_semantics": (
            "sha256(ledger_values_sha256 NUL ledger_availability_sha256)"
        ),
        "shared_core_version": ledger.M1_INPUT_SCHEMA_VERSION,
        "materialized_primary_extractor": (
            "advantage_m1_primary_from_materialized_state"
        ),
        "canonical_primary_extractor": "advantage_m1_primary_from_canonical",
        "shared_tensorizer_function": "tensorize_primary",
        "shared_core_module_sha256": file_sha256(
            REPO_ROOT / "src/advantage_m1_causal_ledger_v3.py"
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    """V6 event表を読み、M0/M1 V3が共有する固定行集合を新規保存する。"""

    if args.output_root.exists():
        raise AdvantageM1DatasetV2Error(f"出力先は新規必須です: {args.output_root}")
    sources = _pilot_sources(args.index_root, args.increment_manifest)
    projected = _projected_index(args.projected_root)
    pairs = [_source_arrays(source, args.table_roots, projected) for source in sources]
    arrays = _concatenate([pair[0] for pair in pairs])
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    _write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(args, sources, arrays, [pair[1] for pair in pairs], dataset_path,
                         projected.receipt)
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
    dataset_path: Path, projected_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    gates = Counter(str(value) for value in arrays["projected_gate_status"])
    return {
        "format_version": FORMAT_VERSION, "not_production": True,
        "source_count": len(sources), "state_count": len(arrays["labels"]),
        "game_count": len(np.unique(arrays["game_keys"])),
        "ledger_usable_count": int(arrays["ledger_usable"].sum()),
        "max_states_per_game": MAX_STATES_PER_GAME,
        "equal_total_weight_per_game": True,
        "source_index_root": str(args.index_root.resolve()),
        "source_index_sha256": file_sha256(args.index_root / "SUMMARY.json"),
        "increment_manifest": (
            None if args.increment_manifest is None
            else {
                "path": str(args.increment_manifest.resolve()),
                "sha256": file_sha256(args.increment_manifest),
            }
        ),
        "table_roots": [str(path.resolve()) for path in args.table_roots],
        "target_ids": [str(row["target_id"]) for row in sources],
        "source_group_ids": [str(row["source_group_id"]) for row in sources],
        "tier_counts": dict(sorted(Counter(str(row["tier"]) for row in sources).items())),
        "training_row_policy": "a_input_usable_and_official_win_label_available",
        "ledger_usable_policy": (
            "b_causal_exchange_usable_and_b_input_usable_no_fault_"
            "and_not_physical_accounting_unsupported_segment"
        ),
        "ledger_policy": "six_primary_fields_finalized_pending_separate_from_provisional",
        "canonical_observation_schema_version": CANONICAL_OBSERVATION_SCHEMA_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
        "m1_input_schema_version": ledger.M1_INPUT_SCHEMA_VERSION,
        "m1_ledger_side_fields": list(ledger.LEDGER_SIDE_FIELDS),
        "shared_tensorizer_full_receipt": _shared_tensorizer_receipt(arrays),
        "state_ids_saved": True,
        "projected_join_role": "audit_and_provenance_only_not_m1_primary_input",
        "projected_input_digest_semantics": "ProjectedStateObservationV1.input_digest",
        "projected_join_counts": dict(sorted(gates.items())),
        "projected_dataset_receipt": dict(projected_receipt),
        "sources": list(audits),
        "code_sha256": {
            "builder": file_sha256(Path(__file__)),
            "tensorizer": file_sha256(REPO_ROOT / "src/advantage_m1_causal_ledger_v3.py"),
            "canonical_schema": file_sha256(REPO_ROOT / "src/canonical_observation_v3.py"),
            "canonical_adapter": file_sha256(
                REPO_ROOT / "src/canonical_observation_adapter_v3.py"
            ),
            "projected_validator": file_sha256(REPO_ROOT / "src/projected_state_dataset_v1.py"),
        },
        "dataset": {"name": dataset_path.name, "sha256": file_sha256(dataset_path)},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--table-roots", type=Path, nargs="+", default=list(DEFAULT_TABLE_ROOTS))
    parser.add_argument("--projected-root", type=Path, default=DEFAULT_PROJECTED_ROOT)
    parser.add_argument("--increment-manifest", type=Path)
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
