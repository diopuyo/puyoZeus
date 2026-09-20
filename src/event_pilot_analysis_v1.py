"""出来事原本v1の短区間・独立試行を比較する読取専用検査。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.event_run_v1 import CompletedRunValidation, validate_completed_run
from src.event_source_v1 import iter_committed_batches


@dataclass(frozen=True, slots=True)
class PilotRunMetrics:
    """一つの完了試行から再計算した容量・順序・内容指標。"""

    run_dir: str
    build_id: str
    attempt_id: str
    event_count: int
    batch_count: int
    part_count: int
    part_bytes: int
    event_json_bytes: int
    event_json_bytes_min: int
    event_json_bytes_max: int
    batch_size_min: int
    batch_size_max: int
    processing_duration_ms: int
    first_available_frame: int
    last_available_frame: int
    semantic_sha256: str
    event_ids_sha256: str
    sequences_sha256: str
    event_type_counts: dict[str, int]


def load_completed_run_events(
    run_dir: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """完了試行を検査し、説明と採用済み出来事を返す。"""

    validation = validate_completed_run(run_dir)
    return load_prevalidated_run_events(run_dir, validation)


def load_prevalidated_run_events(
    run_dir: Path, validation: CompletedRunValidation,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """直前に全検査した同じ完成試行を、再検査せず一度だけ読み込む。"""

    if not validation.valid:
        raise ValueError(f"完了試行が不正です: {run_dir}: {validation.error}")
    manifest = _load_object(run_dir / "manifest.json")
    expected = (validation.event_count, validation.semantic_sha256)
    actual = (manifest.get("adopted_event_count"), manifest.get("semantic_sha256"))
    if actual != expected:
        raise ValueError("直前検査と完成試行マニフェストが一致しません")
    events = tuple(_read_events(run_dir, manifest))
    if len(events) != validation.event_count:
        raise ValueError("直前検査と読込出来事件数が一致しません")
    return manifest, events


def analyze_completed_run(run_dir: Path) -> PilotRunMetrics:
    """完了試行を全再検査して実測値を返す。"""

    manifest, loaded_events = load_completed_run_events(run_dir)
    events = list(loaded_events)
    if not events:
        raise ValueError("完了試行に出来事がありません")
    event_sizes = [_canonical_line_size(event) for event in events]
    batch_sizes = list(Counter(event["availability_batch_id"] for event in events).values())
    parts = manifest["parts"]
    return PilotRunMetrics(
        str(run_dir.resolve()),
        str(manifest["build_id"]),
        str(manifest["attempt_id"]),
        len(events),
        len(batch_sizes),
        len(parts),
        sum(int(part["byte_count"]) for part in parts),
        sum(event_sizes),
        min(event_sizes),
        max(event_sizes),
        min(batch_sizes),
        max(batch_sizes),
        _processing_duration_ms(manifest["source"]),
        int(events[0]["timing"]["available_frame"]),
        int(events[-1]["timing"]["available_frame"]),
        str(manifest["semantic_sha256"]),
        _value_sha256([event["event_id"] for event in events]),
        _value_sha256([event["seq"] for event in events]),
        dict(sorted(Counter(str(event["event_type"]) for event in events).items())),
    )


def compare_completed_runs(run_dirs: list[Path]) -> dict[str, Any]:
    """同一生成内容の独立試行一致を比較する。"""

    if len(run_dirs) < 2:
        raise ValueError("独立試行比較には2件以上必要です")
    metrics = [analyze_completed_run(path) for path in run_dirs]
    if len({item.run_dir for item in metrics}) != len(metrics):
        raise ValueError("同じ実行試行ディレクトリを重複指定できません")
    if len({item.attempt_id for item in metrics}) != len(metrics):
        raise ValueError("独立試行の実行試行IDが重複しています")
    compared = {
        "build_id": [item.build_id for item in metrics],
        "event_count": [item.event_count for item in metrics],
        "semantic_sha256": [item.semantic_sha256 for item in metrics],
        "event_ids_sha256": [item.event_ids_sha256 for item in metrics],
        "sequences_sha256": [item.sequences_sha256 for item in metrics],
    }
    matches = {name: len(set(values)) == 1 for name, values in compared.items()}
    return {
        "attempts_match": all(matches.values()),
        "match_by_field": matches,
        "compared_values": compared,
        "attempt_ids": [item.attempt_id for item in metrics],
        "runs": [asdict(item) for item in metrics],
    }


def _read_events(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    expected_seq = 0
    for part in manifest["parts"]:
        path = run_dir / str(part["relative_path"])
        for batch in iter_committed_batches(path, expected_first_seq=expected_seq):
            batch_events = [dict(event) for event in batch.events]
            events.extend(batch_events)
            expected_seq = int(batch_events[-1]["seq"]) + 1
    return events


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSONオブジェクトではありません: {path}")
    return value


def _canonical_line_size(value: dict[str, Any]) -> int:
    return len(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )


def _value_sha256(value: list[Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _processing_duration_ms(source: dict[str, Any]) -> int:
    frames = int(source["processing_end_frame_exclusive"]) - int(
        source["processing_start_frame"]
    )
    numerator = int(source["time_base_numerator"])
    denominator = int(source["time_base_denominator"])
    return frames * numerator * 1000 // denominator


__all__ = [
    "PilotRunMetrics",
    "analyze_completed_run",
    "compare_completed_runs",
    "load_completed_run_events",
]
