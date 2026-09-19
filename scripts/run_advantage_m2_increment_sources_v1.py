"""M2純増S級2 sourceを一動画ずつevent収集・検証する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import run_event_pilot_48_v1 as event_campaign
from scripts import train_advantage_m2_projected_safe_v1 as m2_trainer


FORMAT_VERSION = "advantage-m2-increment-event-campaign/v1"
COMPLETE_VERSION = "advantage-m2-increment-event-campaign-complete/v1"
ATTEMPT_NAMESPACE = "m2-increment-s1-s2-v1"
DEFAULT_MANIFEST = Path(
    "docs/manifests/ADVANTAGE_M2_INCREMENT_S1_S2_2026-09-06.tsv"
)
DEFAULT_PARENT_MANIFEST = Path(
    "data/verify/advantage_m1_canonical_dataset_46v_2026-09-05_v3_finalized_primary6/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/event_source_v1_m2_increment_s1_s2_2026-09-06_v1"
)
EXPECTED_TARGETS = ("c82", "c83")
EXPECTED_FOLDS = (1, 3)
MANIFEST_FIELDS = (
    "format_version", "target_id", "canonical_video_alias", "historical_alias",
    "youtube_id", "tier", "fold", "video_path", "video_sha256",
    "board_npz_path", "board_npz_sha256", "score_calibration_path",
    "score_calibration_sha256", "game_count", "row_count", "role",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CommandRunner = Callable[[tuple[str, ...], Path, Path], int]


class AdvantageM2IncrementCampaignError(RuntimeError):
    """純増sourceの入力・隔離・収集契約違反。"""


@dataclass(frozen=True, slots=True)
class IncrementSourceV1:
    """投入前に固定した一source。"""

    target_id: str
    video_alias: str
    historical_alias: str
    youtube_id: str
    tier: str
    fold: int
    video_path: Path
    video_sha256: str
    board_path: Path
    board_sha256: str
    calibration_path: Path
    calibration_sha256: str
    game_count: int
    row_count: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdvantageM2IncrementCampaignError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise AdvantageM2IncrementCampaignError(f"JSON objectではありません: {path}")
    return value


def _source(row: Mapping[str, str]) -> IncrementSourceV1:
    if row.get("format_version") != "advantage-m2-increment-source/v1":
        raise AdvantageM2IncrementCampaignError("投入manifest versionが不正です")
    try:
        return IncrementSourceV1(
            target_id=row["target_id"], video_alias=row["canonical_video_alias"],
            historical_alias=row["historical_alias"], youtube_id=row["youtube_id"],
            tier=row["tier"], fold=int(row["fold"]), video_path=Path(row["video_path"]),
            video_sha256=row["video_sha256"], board_path=Path(row["board_npz_path"]),
            board_sha256=row["board_npz_sha256"],
            calibration_path=Path(row["score_calibration_path"]),
            calibration_sha256=row["score_calibration_sha256"],
            game_count=int(row["game_count"]),
            row_count=int(row["row_count"]),
        )
    except (KeyError, ValueError) as error:
        raise AdvantageM2IncrementCampaignError("投入manifest値が不正です") from error


def load_manifest(path: Path) -> tuple[IncrementSourceV1, ...]:
    """固定2行manifestを読み、aliasとfoldの水増し防止を検証する。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise AdvantageM2IncrementCampaignError("投入manifest列が固定仕様と不一致です")
        rows = tuple(_source(row) for row in reader)
    checks = (
        tuple(row.target_id for row in rows) == EXPECTED_TARGETS,
        tuple(row.fold for row in rows) == EXPECTED_FOLDS,
        len({row.youtube_id for row in rows}) == len(rows),
        all(row.tier == "S級" for row in rows),
        all(row.video_path.stem == row.video_alias for row in rows),
        all(row.historical_alias == f"video_{row.target_id}" for row in rows),
    )
    if not all(checks):
        raise AdvantageM2IncrementCampaignError("alias・tier・fold固定契約が不正です")
    return rows


def _validate_asset(root: Path, path: Path, expected_sha256: str, label: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AdvantageM2IncrementCampaignError(f"{label}がありません: {path}")
    if receipt.file_sha256(resolved) != expected_sha256:
        raise AdvantageM2IncrementCampaignError(f"{label}のSHA-256が不一致です: {path}")
    return resolved


def _validate_calibration(path: Path, row: IncrementSourceV1) -> None:
    value = _read_json(path)
    checks = (
        value.get("schema_version") == "score_region_calibration_v1",
        value.get("source_video_id") == row.video_alias,
        value.get("source_video_sha256") == row.video_sha256,
    )
    if not all(checks):
        raise AdvantageM2IncrementCampaignError(f"較正と動画が不一致です: {row.target_id}")


def validate_inputs(
    rows: Sequence[IncrementSourceV1], parent_manifest_path: Path,
) -> dict[str, Any]:
    """既存46-sourceとの非重複と全入力hashを収集前に確定する。"""

    parent = _read_json(parent_manifest_path)
    existing = set(str(value) for value in parent.get("source_group_ids", ()))
    overlap = sorted(existing & {row.youtube_id for row in rows})
    if overlap:
        raise AdvantageM2IncrementCampaignError(f"現行46-sourceと重複しています: {overlap}")
    assets = []
    for row in rows:
        video = _validate_asset(REPO_ROOT, row.video_path, row.video_sha256, "動画")
        board = _validate_asset(REPO_ROOT, row.board_path, row.board_sha256, "盤面NPZ")
        calibration = _validate_asset(
            REPO_ROOT, row.calibration_path, row.calibration_sha256, "スコア領域較正",
        )
        _validate_calibration(calibration, row)
        assets.append({"target_id": row.target_id, "video": str(video),
                       "board": str(board), "calibration": str(calibration)})
    if not production_compatible(REPO_ROOT):
        raise AdvantageM2IncrementCampaignError("production config hashが固定値と不一致です")
    return {"parent_source_count": len(existing), "overlap": overlap, "assets": assets}


def _retry_paths(root: Path, row: IncrementSourceV1) -> event_campaign.RetryPaths:
    attempt = f"{ATTEMPT_NAMESPACE}-{row.target_id}"
    base = root / "targets" / f"target={row.target_id}" / f"attempt={attempt}" / "retry=0000"
    paths = event_campaign.RetryPaths(
        0, attempt, base, base / "work/collection.npz",
        base / "logs/collection.log", base / "results/run.json",
        base / "event-runs", base / "validation",
    )
    for directory in (
        paths.work_npz.parent, paths.collection_log.parent,
        paths.run_result.parent, paths.event_runs, paths.validation_root,
    ):
        directory.mkdir(parents=True, exist_ok=False)
    return paths


def _runner_command(
    row: IncrementSourceV1, paths: event_campaign.RetryPaths,
) -> tuple[tuple[str, ...], event_campaign.VideoMetadata]:
    video = (REPO_ROOT / row.video_path).resolve()
    metadata = event_campaign.probe_video(video)
    max_sec = event_campaign.full_length_max_sec(metadata)
    command = (
        sys.executable, "-u", "-m", "scripts.run_event_snapshot_pilot_v1",
        "--video", str(video), "--work-npz", str(paths.work_npz),
        "--log", str(paths.collection_log), "--output-root", str(paths.event_runs),
        "--attempt-id", paths.attempt_id, "--start-sec", "0",
        "--max-sec", repr(max_sec), "--result-json", str(paths.run_result),
        "--score-region-calibration", str((REPO_ROOT / row.calibration_path).resolve()),
    )
    return command, metadata


def _load_run_reference(
    paths: event_campaign.RetryPaths, metadata: event_campaign.VideoMetadata,
) -> event_campaign.RunReference:
    result = _read_json(paths.run_result)
    requested = result.get("requested_end_frame_exclusive")
    processed = result.get("processed_end_frame_exclusive")
    if requested != metadata.frame_count or processed != metadata.frame_count:
        raise AdvantageM2IncrementCampaignError("全長処理frame数が動画と一致しません")
    run_text = result.get("run_dir")
    run_dir = Path(run_text).resolve() if isinstance(run_text, str) else Path()
    if not run_dir.is_relative_to(paths.event_runs.resolve()):
        raise AdvantageM2IncrementCampaignError("event runが隔離root外です")
    if any(not (run_dir / marker).is_file() for marker in event_campaign.RUN_MARKERS):
        raise AdvantageM2IncrementCampaignError("event runの完了receiptが不足しています")
    return event_campaign.RunReference(run_dir, paths)


def _process_source(
    root: Path, row: IncrementSourceV1, command_runner: CommandRunner,
) -> dict[str, Any]:
    paths = _retry_paths(root, row)
    command, metadata = _runner_command(row, paths)
    runner_log = paths.root / "logs/runner.log"
    code = command_runner(command, REPO_ROOT, runner_log)
    if code != 0:
        raise AdvantageM2IncrementCampaignError(f"収集に失敗しました: {row.target_id}")
    run = _load_run_reference(paths, metadata)
    validation = event_campaign.run_validations(run, command_runner)
    if not validation.all_pass:
        raise AdvantageM2IncrementCampaignError(f"event品質gate不合格です: {row.target_id}")
    return {
        "target_id": row.target_id, "youtube_id": row.youtube_id,
        "fold": row.fold, "run_dir": str(run.run_dir),
        "validation": validation.summary,
    }


def _safe_process(
    root: Path, row: IncrementSourceV1, command_runner: CommandRunner,
) -> dict[str, Any]:
    try:
        value = _process_source(root, row, command_runner)
        return {"state": "completed", **value}
    except Exception as error:
        failure = {"state": "failed", "target_id": row.target_id,
                   "youtube_id": row.youtube_id, "error": repr(error)}
        path = root / "targets" / f"target={row.target_id}" / "FAILURE.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(path, failure)
        return failure


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_campaign(
    args: argparse.Namespace, command_runner: CommandRunner = event_campaign._execute_command,
) -> dict[str, Any]:
    """2 sourceを順次処理し、失敗成果も消さずに最終receiptを残す。"""

    if args.output_root.exists():
        raise AdvantageM2IncrementCampaignError(f"出力先は新規必須です: {args.output_root}")
    rows = load_manifest(args.manifest)
    preflight = validate_inputs(rows, args.parent_manifest)
    if getattr(args, "preflight_only", False):
        return {
            "format_version": FORMAT_VERSION, "not_production": True,
            "preflight": preflight, "source_count": len(rows),
            "all_pass": True, "collection_started": False,
        }
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [_safe_process(args.output_root, row, command_runner) for row in rows]
    all_pass = all(row["state"] == "completed" for row in results)
    report = {
        "format_version": FORMAT_VERSION, "not_production": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": receipt.file_sha256(args.manifest),
        "preflight": preflight, "worker_count": 1, "results": results,
        "all_pass": all_pass, "video_deleted": False,
        "production_config_changed": False,
    }
    report_path = args.output_root / "SUMMARY.json"
    _write_json_exclusive(report_path, report)
    finished = {"format_version": COMPLETE_VERSION, "all_pass": all_pass,
                "summary_sha256": receipt.file_sha256(report_path)}
    _write_json_exclusive(args.output_root / "RUN_FINISHED", finished)
    if all_pass:
        _write_json_exclusive(args.output_root / "COMPLETE", finished)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_campaign(parse_args(argv))
    print(json.dumps({"all_pass": report["all_pass"]}, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
