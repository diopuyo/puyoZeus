"""親M1 datasetの物理会計quarantine行だけを厳密M0 fallbackへ直す。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts import build_advantage_m1_dataset_v2 as builder
from scripts import train_advantage_m1_zero_counterfactual_v3 as trainer
from src import advantage_m1_causal_ledger_v3 as ledger


FORMAT_VERSION = "advantage-m1-training-quarantine-sanitizer/v1"
QUALITY_FLAG = "quality_physical_accounting_unsupported_segment"
EXPECTED_PRODUCTION_CONFIG_SHA256 = (
    "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
)
LEDGER_ARRAYS = ("ledger_values", "ledger_availability", "ledger_usable")
REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingQuarantineSanitizerError(RuntimeError):
    """親dataset、学習表、または派生datasetの固定契約違反。"""


def _parent_arrays(samples: trainer.CanonicalSamplesV3) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(samples, name))
        for name in trainer.CanonicalSamplesV3.__dataclass_fields__
        if name != "folds"
    }


def _load_parent(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        samples, manifest = trainer.load_canonical_dataset_v2(root)
    except (OSError, ValueError, RuntimeError) as error:
        raise TrainingQuarantineSanitizerError("親datasetの検証に失敗しました") from error
    return _parent_arrays(samples), manifest


def _validate_source(source: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(str(source.get("table_root", "")))
    manifest_path, complete_path = root / "manifest.json", root / "COMPLETE"
    manifest = builder._load_json(manifest_path)
    complete = builder._load_json(complete_path)
    if builder.file_sha256(manifest_path) != source.get("table_manifest_sha256"):
        raise TrainingQuarantineSanitizerError(f"親source manifestが変化しました: {root}")
    if complete.get("manifest_sha256") != builder.file_sha256(manifest_path):
        raise TrainingQuarantineSanitizerError(f"source COMPLETEが不正です: {root}")
    for name in ("states", "labels"):
        entry = manifest.get("tables", {}).get(name, {})
        # manifest名は固定表名と厳密一致のみ許可（別名・別dir・絶対path・親参照を救済しない）
        if entry.get("name") != f"{name}.parquet":
            raise TrainingQuarantineSanitizerError(
                f"source table名が固定名ではありません: {root}/{name}"
            )
        path = root / f"{name}.parquet"
        if not path.is_file() or entry.get("sha256") != builder.file_sha256(path):
            raise TrainingQuarantineSanitizerError(f"source tableが変化しました: {path}")
    return root, manifest


def _read_states(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    try:
        return parquet.read_table(root / "states.parquet").to_pylist()
    except (OSError, ValueError) as error:
        raise TrainingQuarantineSanitizerError(f"statesを読めません: {root}") from error


def _replace_quarantined_row(
    arrays: dict[str, np.ndarray], position: int, state: Mapping[str, Any],
) -> bool:
    inputs = ledger.tensorize_primary(
        ledger.advantage_m1_primary_from_materialized_state(state),
    )
    if not np.array_equal(inputs.boards, arrays["boards"][position]):
        raise TrainingQuarantineSanitizerError("親datasetと盤面tensorが一致しません")
    if not np.array_equal(inputs.queues, arrays["queues"][position]):
        raise TrainingQuarantineSanitizerError("親datasetとqueue tensorが一致しません")
    before = bool(arrays["ledger_usable"][position])
    arrays["ledger_values"][position] = inputs.ledger_values
    arrays["ledger_availability"][position] = inputs.ledger_availability
    arrays["ledger_usable"][position] = ledger.materialized_ledger_is_usable(state)
    if bool(arrays["ledger_usable"][position]):
        raise TrainingQuarantineSanitizerError("quarantine行がM1 usableのままです")
    return before


def _join_source(
    arrays: dict[str, np.ndarray], index: Mapping[str, int], seen: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    quarantined = closed = 0
    for state in rows:
        position = index.get(str(state.get("state_id", "")))
        if position is None:
            continue
        if seen[position]:
            raise TrainingQuarantineSanitizerError("state_idが複数sourceへjoinしました")
        seen[position] = True
        if state.get(QUALITY_FLAG) is True:
            quarantined += 1
            closed += int(_replace_quarantined_row(arrays, position, state))
    return quarantined, closed


def _sanitize(
    arrays: dict[str, np.ndarray], sources: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    index = {str(value): row for row, value in enumerate(arrays["state_ids"])}
    seen = np.zeros(len(index), dtype=np.bool_)
    audits: list[dict[str, Any]] = []
    totals = {"quarantined_state_count": 0, "closed_leak_count": 0}
    for source in sources:
        root, manifest = _validate_source(source)
        quarantined, closed = _join_source(arrays, index, seen, _read_states(root))
        audits.append({
            "target_id": str(source["target_id"]),
            "table_root": str(root.resolve()),
            "table_manifest_sha256": builder.file_sha256(root / "manifest.json"),
            "quarantined_state_count": quarantined, "closed_leak_count": closed,
            "partition_fold": int(manifest["partition_fold"]),
        })
        totals["quarantined_state_count"] += quarantined
        totals["closed_leak_count"] += closed
    if not bool(seen.all()):
        raise TrainingQuarantineSanitizerError(f"source join漏れ: {int((~seen).sum())}行")
    return audits, totals


def _unchanged_receipt(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray],
) -> dict[str, str]:
    names = sorted(set(before) - set(LEDGER_ARRAYS))
    result: dict[str, str] = {}
    for name in names:
        old_hash = builder._array_sha256(before[name])
        new_hash = builder._array_sha256(after[name])
        if old_hash != new_hash:
            raise TrainingQuarantineSanitizerError(f"非ledger列が変化しました: {name}")
        result[name] = new_hash
    return result


def _source_usable_counts(
    arrays: Mapping[str, np.ndarray], groups: Sequence[str],
) -> dict[str, int]:
    return {
        str(group): int(arrays["ledger_usable"][arrays["source_groups"] == group].sum())
        for group in groups
    }


def _manifest(
    args: argparse.Namespace, arrays: Mapping[str, np.ndarray],
    parent: Mapping[str, Any], audits: Sequence[Mapping[str, Any]],
    totals: Mapping[str, int], unchanged: Mapping[str, str], dataset_path: Path,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(parent))
    groups = [str(value) for value in output["source_group_ids"]]
    usable = _source_usable_counts(arrays, groups)
    for source, group in zip(output["sources"], groups, strict=True):
        source["ledger_usable_count"] = usable[group]
    output["ledger_usable_count"] = int(arrays["ledger_usable"].sum())
    output["ledger_usable_policy"] = (
        "b_causal_exchange_usable_and_b_input_usable_no_fault_"
        "and_not_physical_accounting_unsupported_segment"
    )
    output["shared_tensorizer_full_receipt"] = builder._shared_tensorizer_receipt(arrays)
    output["quarantine_sanitizer"] = _lineage(
        args, audits, totals, unchanged,
    )
    output["code_sha256"]["tensorizer"] = builder.file_sha256(
        REPO_ROOT / "src/advantage_m1_causal_ledger_v3.py",
    )
    output["code_sha256"]["quarantine_sanitizer"] = builder.file_sha256(Path(__file__))
    output["dataset"] = {"name": dataset_path.name, "sha256": builder.file_sha256(dataset_path)}
    return output


def _lineage(
    args: argparse.Namespace, audits: Sequence[Mapping[str, Any]],
    totals: Mapping[str, int], unchanged: Mapping[str, str],
) -> dict[str, Any]:
    root = args.parent_root
    return {
        "format_version": FORMAT_VERSION, "quality_flag": QUALITY_FLAG,
        "parent_dataset_root": str(root.resolve()),
        "parent_manifest_sha256": builder.file_sha256(root / "manifest.json"),
        "parent_complete_sha256": builder.file_sha256(root / "COMPLETE"),
        "parent_dataset_sha256": builder.file_sha256(root / "dataset.npz"),
        "projected_columns_reused_without_rescan": True,
        "production_config_sha256": builder.file_sha256(
            REPO_ROOT / "src/production_config.py",
        ),
        "quarantined_state_count": int(totals["quarantined_state_count"]),
        "closed_leak_count": int(totals["closed_leak_count"]),
        "unchanged_array_sha256": dict(unchanged), "sources": list(audits),
    }


def _validate_production_config() -> None:
    if not production_compatible(REPO_ROOT):
        raise TrainingQuarantineSanitizerError("production依存値が事前登録時から変化しました")


def build(args: argparse.Namespace) -> dict[str, Any]:
    """親datasetの行・fold・M0列を固定し、M1 quarantineだけを派生保存する。"""

    if args.output_root.exists():
        raise TrainingQuarantineSanitizerError(f"出力先は新規必須です: {args.output_root}")
    _validate_production_config()
    parent_arrays, parent_manifest = _load_parent(args.parent_root)
    arrays = {name: value.copy() for name, value in parent_arrays.items()}
    audits, totals = _sanitize(arrays, parent_manifest["sources"])
    unchanged = _unchanged_receipt(parent_arrays, arrays)
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_root / "dataset.npz"
    builder._write_npz_exclusive(dataset_path, arrays)
    manifest = _manifest(
        args, arrays, parent_manifest, audits, totals, unchanged, dataset_path,
    )
    manifest_path = args.output_root / "manifest.json"
    builder._write_json_exclusive(manifest_path, manifest)
    builder._write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": builder.COMPLETE_VERSION,
        "manifest_sha256": builder.file_sha256(manifest_path),
        "dataset_sha256": builder.file_sha256(dataset_path),
    })
    trainer.load_canonical_dataset_v2(args.output_root)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps({
        "source_count": manifest["source_count"],
        "state_count": manifest["state_count"],
        "ledger_usable_count": manifest["ledger_usable_count"],
        "quarantine_sanitizer": manifest["quarantine_sanitizer"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
