"""M1 training quarantine sanitizerの回帰テスト。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import sanitize_advantage_m1_training_quarantine_v1 as subject
from src import advantage_m1_causal_ledger_v3 as ledger
from src.canonical_observation_v3 import AvailabilityState
from tests.test_advantage_m1_causal_ledger_v3 import _state


def _arrays(state: dict[str, object]) -> dict[str, np.ndarray]:
    inputs = ledger.advantage_m1_inputs_from_materialized_state(state)
    return {
        "boards": inputs.boards[None].copy(),
        "queues": inputs.queues[None].copy(),
        "ledger_values": inputs.ledger_values[None].copy(),
        "ledger_availability": inputs.ledger_availability[None].copy(),
        "ledger_usable": np.asarray([True], dtype=np.bool_),
        "state_ids": np.asarray(["s1"]),
    }


def test_join_source_closes_quarantined_usable_row() -> None:
    clean = _state()
    arrays = _arrays(clean)
    quarantined = clean | {
        "state_id": "s1",
        subject.QUALITY_FLAG: True,
    }
    seen = np.zeros(1, dtype=np.bool_)

    count, closed = subject._join_source(arrays, {"s1": 0}, seen, [quarantined])

    unsupported = list(AvailabilityState).index(AvailabilityState.UNSUPPORTED)
    assert (count, closed) == (1, 1)
    assert seen.tolist() == [True]
    assert not arrays["ledger_usable"][0]
    np.testing.assert_array_equal(arrays["ledger_values"][0, :, 0], 0.0)
    np.testing.assert_array_equal(
        arrays["ledger_availability"][0, :, 0, unsupported], 1.0,
    )


def test_join_source_does_not_rewrite_unflagged_row() -> None:
    state = _state() | {"state_id": "s1"}
    arrays = _arrays(state)
    before = {name: value.copy() for name, value in arrays.items()}

    count, closed = subject._join_source(
        arrays, {"s1": 0}, np.zeros(1, dtype=np.bool_), [state],
    )

    assert (count, closed) == (0, 0)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])


def test_unchanged_receipt_rejects_m0_column_change() -> None:
    state = _state()
    before, after = _arrays(state), _arrays(state)
    after["boards"][0, 0, 0, 0] += 1

    with np.testing.assert_raises(subject.TrainingQuarantineSanitizerError):
        subject._unchanged_receipt(before, after)


# (case名, 対象table, manifestに書く別名(Noneはname欠落, <abs>は絶対path), 固定名を消すか)
_ALIAS_CASES: tuple[tuple[str, str, str | None, bool], ...] = (
    ("states_alias", "states", "trusted_states.parquet", False),
    ("labels_alias", "labels", "trusted_labels.parquet", False),
    ("states_subdir", "states", "nested/states.parquet", False),
    ("states_parent", "states", "../states.parquet", False),
    ("states_absolute", "states", "<abs>", False),
    ("states_missing_name", "states", None, False),
    ("labels_alias_only", "labels", "trusted_labels.parquet", True),
)


def _fixed_manifest(root: Path) -> dict[str, Any]:
    """固定名states/labelsを書き、その正SHAを持つmanifest雛形を返す。"""
    for name in ("states", "labels"):
        (root / f"{name}.parquet").write_bytes(f"fixed-{name}".encode("ascii"))
    return {
        "tables": {
            name: {
                "name": f"{name}.parquet",
                "sha256": subject.builder.file_sha256(root / f"{name}.parquet"),
            } for name in ("states", "labels")
        },
    }


def _seal_source(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """manifest/COMPLETEを整合させ、対応する親source記録を返す。"""
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = subject.builder.file_sha256(root / "manifest.json")
    (root / "COMPLETE").write_text(
        json.dumps({"manifest_sha256": manifest_sha}), encoding="utf-8",
    )
    return {"table_root": str(root), "table_manifest_sha256": manifest_sha}


def _mutate(root: Path, table: str, alias: str | None, drop_fixed: bool) -> dict[str, Any]:
    """固定名rootへ別名entryを注入し、必要なら固定名tableを取り除く。"""
    manifest = _fixed_manifest(root)
    if alias is None:
        manifest["tables"][table] = {"sha256": "0" * 64}
        return manifest
    if alias == "<abs>":
        alias = str(root / f"{table}.parquet")
    target = Path(alias) if Path(alias).is_absolute() else root / alias
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(f"alias-{table}".encode("ascii"))
    manifest["tables"][table] = {
        "name": alias, "sha256": subject.builder.file_sha256(target),
    }
    if drop_fixed:
        (root / f"{table}.parquet").unlink()
    return manifest


def test_non_fixed_table_names_are_rejected_before_states_read(tmp_path: Path) -> None:
    """sanitizerも別名manifestを_read_states到達前に拒否する。"""
    for case, table, alias, drop_fixed in _ALIAS_CASES:
        root = tmp_path / case
        root.mkdir()
        source = _seal_source(root, _mutate(root, table, alias, drop_fixed))

        with np.testing.assert_raises(subject.TrainingQuarantineSanitizerError):
            subject._validate_source(source)


def test_fixed_names_pass_and_stale_fixed_sha_still_fails(tmp_path: Path) -> None:
    """正常な固定名は従来どおり通り、固定名のSHA不一致は引き続き拒否する。"""
    root = tmp_path / "ok"
    root.mkdir()
    source = _seal_source(root, _fixed_manifest(root))

    validated_root, manifest = subject._validate_source(source)

    assert validated_root == root
    assert manifest["tables"]["states"]["name"] == "states.parquet"
    stale = tmp_path / "stale"
    stale.mkdir()
    stale_source = _seal_source(stale, _fixed_manifest(stale))
    (stale / "states.parquet").write_bytes(b"rewritten-after-manifest")
    with np.testing.assert_raises(subject.TrainingQuarantineSanitizerError):
        subject._validate_source(stale_source)
