"""検収済みNPZと観測サイドカーを現在版の原本へ新規再出力する。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.run_event_snapshot_pilot_v1 import (
    build_source_spec,
    probe_video,
    runtime_environment_spec,
)
from src.event_pilot_analysis_v1 import load_completed_run_events
from src.event_run_v1 import SourceVideoSpec
from src.event_snapshot_export_v1 import (
    SnapshotExportRequest,
    discover_runtime_artifacts,
    export_snapshot_run,
    file_sha256,
)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """旧完了試行を根拠に、入力を再認識せず新しい原本だけを作る。"""

    project_root = Path(__file__).resolve().parent.parent
    source, old_config, base_input, source_already_hashed = _load_base_input(
        args, project_root,
    )
    video_path = project_root / source.source_video_path
    if not source_already_hashed:
        _verify_source(video_path, source.source_video_sha256)
    per_video_relative = _per_video_config_relative(old_config)
    per_video_config = (
        project_root / per_video_relative if per_video_relative is not None else None
    )
    artifacts = discover_runtime_artifacts(project_root, per_video_config)
    request = SnapshotExportRequest(
        args.npz,
        args.output_root,
        args.attempt_id,
        source,
        tuple(str(token) for token in old_config["collection_tokens"]),
        artifacts,
        runtime_environment_spec(),
        per_video_relative,
        event_observation_sidecar_path=args.observation_sidecar,
        event_accounting_sidecar_path=args.accounting_sidecar,
        event_physical_sidecar_path=args.physical_sidecar,
    )
    result = export_snapshot_run(request)
    _verify_source(video_path, source.source_video_sha256)
    if discover_runtime_artifacts(project_root, per_video_config) != artifacts:
        raise RuntimeError("再出力中に対象コード・認識資産が変わりました")
    return {
        "base_input": base_input,
        "run_dir": str(result.run_dir.resolve()),
        "build_id": result.build_id,
        "event_count": result.event_count,
        "semantic_sha256": result.semantic_sha256,
        "stable_board_event_count": result.observed_row_count,
        "boundary_event_count": result.boundary_event_count,
        "winner_observed_event_count": result.winner_observed_event_count,
        "official_game_assignment_event_count": result.official_game_assignment_event_count,
        "excluded_winner_result_counts": dict(result.excluded_winner_result_counts),
        "input_npz_sha256": result.input_npz_sha256,
        "observation_sidecar_sha256": result.observation_sidecar_sha256,
        "accounting_sidecar_sha256": result.accounting_sidecar_sha256,
        "accounting_inspected_side_count": result.accounting_inspected_side_count,
        "accounting_nonzero_row_count": result.accounting_nonzero_row_count,
        "accounting_event_type_counts": dict(result.accounting_event_type_counts),
        "physical_sidecar_sha256": result.physical_sidecar_sha256,
        "physical_inspected_side_count": result.physical_inspected_side_count,
        "physical_row_count": result.physical_row_count,
        "physical_event_type_counts": dict(result.physical_event_type_counts),
    }


def _load_base_input(
    args: argparse.Namespace, project_root: Path,
) -> tuple[SourceVideoSpec, dict[str, Any], str, bool]:
    if args.base_run_dir is not None:
        manifest, _ = load_completed_run_events(args.base_run_dir)
        config_path = args.base_run_dir / "recognition-config.json"
        return (
            SourceVideoSpec(**manifest["source"]),
            _load_object(config_path),
            str(args.base_run_dir.resolve()),
            False,
        )
    if args.base_config is None or args.video is None:
        raise ValueError("未完了試行の再出力にはbase-configとvideoが必要です")
    config = _load_object(args.base_config)
    source = _source_from_failed_config(project_root, args.video, config)
    return source, config, str(args.base_config.resolve()), True


def _source_from_failed_config(
    project_root: Path, video: Path, config: dict[str, Any],
) -> SourceVideoSpec:
    resolved = video.resolve()
    metadata = probe_video(resolved)
    source_id = str(config.get("source_video_id", ""))
    if source_id != resolved.stem:
        raise ValueError("基準設定の映像IDが指定映像と一致しません")
    start = _config_int(config, "processing_start_frame")
    end = _config_int(config, "processing_end_frame_exclusive")
    source = build_source_spec(
        project_root, resolved, metadata, file_sha256(resolved), start, end,
    )
    if (source.time_base_numerator != _config_int(config, "time_base_numerator")
            or source.time_base_denominator != _config_int(
                config, "time_base_denominator",
            )):
        raise ValueError("基準設定の時間単位が指定映像と一致しません")
    return source


def _config_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"基準設定の{name}が整数ではありません")
    return value


def _per_video_config_relative(config: dict[str, Any]) -> str | None:
    value = config.get("per_video_config_relative_path")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("映像別設定の相対位置が不正です")
    return value


def _verify_source(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise RuntimeError("元映像が検収済み完了試行と一致しません")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("collection_tokens"), list):
        raise ValueError("基準試行の認識設定が不正です")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base-run-dir", type=Path)
    base.add_argument("--base-config", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--observation-sidecar", type=Path, required=True)
    parser.add_argument("--accounting-sidecar", type=Path)
    parser.add_argument("--physical-sidecar", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.base_config is not None and args.video is None:
        parser.error("base-configを使う場合はvideoが必要です")
    return args


def main() -> int:
    args = parse_args()
    result = execute(args)
    payload = (
        json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    if args.result_json is not None:
        _write_exclusive(args.result_json, payload)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
