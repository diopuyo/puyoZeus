"""固定pilot 48本を読取り専用入力として全長収集し、動画は移動・削除しない。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "docs/manifests/EVENT_PILOT_2026-08-28.tsv"
CALIBRATION_DIR = (
    PROJECT_ROOT / "data/verify/score_region_calibration_48pilot_2026-08-29"
)
EXPECTED_ALL_COUNT = 97
EXPECTED_PILOT_COUNT = 48
EXPECTED_RESERVE_COUNT = 49
MIN_WORKERS = 1
MAX_WORKERS = 3
DEFAULT_WORKERS = 3
FULL_LENGTH_SAFETY_SECONDS = 60.0
ATTEMPT_NAMESPACE = "event-pilot-48-v1"
STATUS_FORMAT = "event-pilot-48-status/1"
SUMMARY_FORMAT = "event-pilot-48-summary/1"
VALIDATION_FORMAT = "event-pilot-48-validation/1"
RUN_MARKERS = ("manifest.json", "validation.json", "COMPLETE")
MANIFEST_FIELDS = (
    "target_id", "subset", "fold", "source_group_id", "source_video_path",
    "clip_start_sec", "clip_end_sec", "tier", "prior_rows",
    "selection_hash", "title",
)
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")
RETRY_PATTERN = re.compile(r"retry=(\d{4})\Z")
ROUND_PATTERN = re.compile(r"round=(\d{4})\Z")


CommandRunner = Callable[[tuple[str, ...], Path, Path], int]
VideoProbe = Callable[[Path], "VideoMetadata"]


@dataclass(frozen=True, slots=True)
class ManifestTarget:
    """固定manifestの一行。"""

    target_id: str
    subset: str
    source_group_id: str
    source_video_path: Path
    clip_start_sec: str
    clip_end_sec: str


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """全長指定に必要な映像情報。"""

    frame_count: int
    fps: float


@dataclass(frozen=True, slots=True)
class VideoPlan:
    """検査済みpilot一件の実行計画。"""

    target: ManifestTarget
    video_path: Path
    calibration_path: Path
    metadata: VideoMetadata
    max_sec: float


@dataclass(frozen=True, slots=True)
class RetryPaths:
    """一回の再試行に閉じた成果物位置。"""

    retry_index: int
    attempt_id: str
    root: Path
    work_npz: Path
    collection_log: Path
    run_result: Path
    event_runs: Path
    validation_root: Path


@dataclass(frozen=True, slots=True)
class ValidationPaths:
    """上書きしない一回の再検証位置。"""

    round_index: int
    root: Path
    logs: Path
    results: Path


@dataclass(frozen=True, slots=True)
class RunReference:
    """全長完了を確認した既存runner成果物。"""

    run_dir: Path
    retry: RetryPaths


@dataclass(frozen=True, slots=True)
class ResumeCandidate:
    """再収集せず検証から再開できる原本。"""

    run: RunReference
    prior_validation_pass: bool


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """3検証器の統合結果。"""

    all_pass: bool
    summary: dict[str, object]
    round_root: Path


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """一動画の終了状態。"""

    target_id: str
    state: str
    retry_index: int | None
    attempt_id: str | None
    run_dir: str | None
    validation: dict[str, object] | None
    error: str | None


class StatusSink:
    """各状態を一行一ファイルの排他JSONLとして保存する。"""

    def __init__(self, output_root: Path) -> None:
        self._root = output_root / "status"
        self._lock = threading.Lock()

    def emit(
        self, target_id: str, state: str, retry_index: int | None,
        detail: Mapping[str, object] | None = None,
    ) -> Path:
        payload: dict[str, object] = {
            "format": STATUS_FORMAT,
            "recorded_unix_ns": time.time_ns(),
            "target_id": target_id,
            "state": state,
            "retry_index": retry_index,
            "detail": dict(detail or {}),
        }
        with self._lock:
            target_root = self._root / f"target={target_id}"
            target_root.mkdir(parents=True, exist_ok=True)
            name = f"status-{time.time_ns()}-{uuid.uuid4().hex}.jsonl"
            path = target_root / name
            _write_bytes_exclusive(path, _canonical_bytes(payload))
        return path


def load_fixed_manifest(
    path: Path = MANIFEST_PATH,
) -> tuple[tuple[ManifestTarget, ...], tuple[ManifestTarget, ...]]:
    """97行の固定manifestを読み、pilot/reserve分離を強制する。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifestの列または列順が固定仕様と一致しません")
        rows = tuple(_manifest_target(row) for row in reader)
    pilot = tuple(row for row in rows if row.subset == "pilot")
    reserve = tuple(row for row in rows if row.subset == "reserve")
    _validate_manifest_rows(rows, pilot, reserve)
    return pilot, reserve


def _manifest_target(row: Mapping[str, str]) -> ManifestTarget:
    target_id = row.get("target_id", "")
    if not target_id or SAFE_ID_PATTERN.fullmatch(target_id) is None:
        raise ValueError(f"安全でないtarget_idです: {target_id!r}")
    return ManifestTarget(
        target_id=target_id,
        subset=row.get("subset", ""),
        source_group_id=row.get("source_group_id", ""),
        source_video_path=Path(row.get("source_video_path", "")),
        clip_start_sec=row.get("clip_start_sec", ""),
        clip_end_sec=row.get("clip_end_sec", ""),
    )


def _validate_manifest_rows(
    rows: Sequence[ManifestTarget], pilot: Sequence[ManifestTarget],
    reserve: Sequence[ManifestTarget],
) -> None:
    counts = (len(rows), len(pilot), len(reserve))
    expected = (EXPECTED_ALL_COUNT, EXPECTED_PILOT_COUNT, EXPECTED_RESERVE_COUNT)
    if counts != expected:
        raise ValueError(f"manifest件数が固定値と不一致です: actual={counts} expected={expected}")
    if any(row.subset not in {"pilot", "reserve"} for row in rows):
        raise ValueError("manifestにpilot/reserve以外のsubsetがあります")
    _require_unique((row.target_id for row in rows), "target_id")
    if any(not row.source_group_id for row in rows):
        raise ValueError("source_group_idは空にできません")
    if any(row.clip_start_sec or row.clip_end_sec for row in pilot):
        raise ValueError("clip指定されたpilotは全長収集対象にできません")
    _validate_expected_video_ids(pilot)
    _validate_split_disjoint(pilot, reserve)


def _validate_expected_video_ids(rows: Sequence[ManifestTarget]) -> None:
    for row in rows:
        expected = f"video_{row.target_id}"
        if row.source_video_path.stem != expected:
            raise ValueError(
                f"target_idと動画IDが一致しません: {row.target_id} / {row.source_video_path.stem}"
            )


def _validate_split_disjoint(
    pilot: Sequence[ManifestTarget], reserve: Sequence[ManifestTarget],
) -> None:
    dimensions = (
        ("target_id", {row.target_id for row in pilot}, {row.target_id for row in reserve}),
        ("source_group_id", {row.source_group_id for row in pilot},
         {row.source_group_id for row in reserve}),
        ("source_video_path", {row.source_video_path.as_posix() for row in pilot},
         {row.source_video_path.as_posix() for row in reserve}),
    )
    for name, pilot_values, reserve_values in dimensions:
        overlap = sorted(pilot_values & reserve_values)
        if overlap:
            raise ValueError(f"pilot/reserveの{name}が交差しています: {overlap}")


def _require_unique(values: Iterable[str], name: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"manifestの{name}が重複しています")


def load_calibration_map(
    calibration_dir: Path, pilot: Sequence[ManifestTarget],
) -> dict[str, Path]:
    """較正JSON集合がpilot動画集合と完全一致することを確認する。"""

    files = tuple(sorted(calibration_dir.glob("*.json")))
    by_video_id: dict[str, Path] = {}
    for path in files:
        video_id = _calibration_video_id(path)
        if video_id in by_video_id:
            raise ValueError(f"較正JSONの動画IDが重複しています: {video_id}")
        by_video_id[video_id] = path.resolve()
    expected = {target.source_video_path.stem for target in pilot}
    actual = set(by_video_id)
    if actual != expected or len(files) != EXPECTED_PILOT_COUNT:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"較正JSON集合がpilotと不一致です: missing={missing} extra={extra}")
    return {target.target_id: by_video_id[target.source_video_path.stem] for target in pilot}


def _calibration_video_id(path: Path) -> str:
    value = _read_json_object(path)
    video_id = value.get("source_video_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"較正JSONのsource_video_idが不正です: {path}")
    return video_id


def probe_video(path: Path) -> VideoMetadata:
    """OpenCVで全長指定に必要なframe_count/fpsを読む。"""

    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"動画のframe_count/fpsを取得できません: {path}")
    return VideoMetadata(frame_count, fps)


def build_video_plans(
    project_root: Path, pilot: Sequence[ManifestTarget],
    calibrations: Mapping[str, Path], probe: VideoProbe = probe_video,
) -> tuple[VideoPlan, ...]:
    """pilot 48本の存在と全長メタデータを先に検査する。"""

    root = project_root.resolve()
    plans: list[VideoPlan] = []
    for target in pilot:
        video = (root / target.source_video_path).resolve()
        if not video.is_relative_to(root) or not video.is_file():
            raise FileNotFoundError(f"pilot動画が存在しません: {target.source_video_path}")
        metadata = probe(video)
        plans.append(VideoPlan(
            target, video, calibrations[target.target_id], metadata,
            full_length_max_sec(metadata),
        ))
    if len(plans) != EXPECTED_PILOT_COUNT:
        raise ValueError("pilot動画計画が48件ではありません")
    return tuple(plans)


def full_length_max_sec(metadata: VideoMetadata) -> float:
    """単体runnerの正数制約を満たし、映像終端を十分超える秒数を返す。"""

    value = metadata.frame_count / metadata.fps + FULL_LENGTH_SAFETY_SECONDS
    if value <= 0 or int(value * metadata.fps) <= metadata.frame_count:
        raise ValueError("全長を超えるmax_secを構成できません")
    return value


def select_pilot_plans(
    plans: Sequence[VideoPlan], requested_ids: Sequence[str] | None,
) -> tuple[VideoPlan, ...]:
    """テスト用target指定をpilot集合の縮小にだけ許可する。"""

    if requested_ids is None:
        return tuple(plans)
    requested = tuple(requested_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--targetsは重複のない1件以上を指定してください")
    pilot_ids = {plan.target.target_id for plan in plans}
    unknown = sorted(set(requested) - pilot_ids)
    if unknown:
        raise ValueError(f"reserveまたは未知targetは指定できません: {unknown}")
    requested_set = set(requested)
    return tuple(plan for plan in plans if plan.target.target_id in requested_set)


def reserve_retry(output_root: Path, target_id: str) -> RetryPaths:
    """既存試行を触らず次のretry番号を排他的に確保する。"""

    parent = _attempt_root(output_root, target_id)
    parent.mkdir(parents=True, exist_ok=True)
    next_index = max((_numbered_name(path, RETRY_PATTERN) for path in parent.iterdir()), default=-1) + 1
    while True:
        root = parent / f"retry={next_index:04d}"
        try:
            root.mkdir(exist_ok=False)
            break
        except FileExistsError:
            next_index += 1
    paths = _retry_paths(root, target_id, next_index)
    for directory in (
        paths.work_npz.parent, paths.collection_log.parent,
        paths.run_result.parent, paths.event_runs, paths.validation_root,
    ):
        directory.mkdir(exist_ok=False)
    return paths


def _attempt_root(output_root: Path, target_id: str) -> Path:
    stable_attempt = f"{ATTEMPT_NAMESPACE}-{target_id}"
    return (
        output_root.resolve() / "targets" / f"target={target_id}"
        / f"attempt={stable_attempt}"
    )


def _retry_paths(root: Path, target_id: str, retry_index: int) -> RetryPaths:
    attempt_id = f"{ATTEMPT_NAMESPACE}-{target_id}-retry-{retry_index:04d}"
    return RetryPaths(
        retry_index, attempt_id, root,
        root / "work" / "collection.npz",
        root / "logs" / "collection.log",
        root / "results" / "run.json",
        root / "event-runs",
        root / "validation",
    )


def reserve_validation_round(retry: RetryPaths) -> ValidationPaths:
    """既存検証を保持したまま次の再検証roundを確保する。"""

    retry.validation_root.mkdir(parents=True, exist_ok=True)
    next_index = max(
        (_numbered_name(path, ROUND_PATTERN) for path in retry.validation_root.iterdir()),
        default=-1,
    ) + 1
    while True:
        root = retry.validation_root / f"round={next_index:04d}"
        try:
            root.mkdir(exist_ok=False)
            break
        except FileExistsError:
            next_index += 1
    logs, results = root / "logs", root / "results"
    logs.mkdir(exist_ok=False)
    results.mkdir(exist_ok=False)
    return ValidationPaths(next_index, root, logs, results)


def _numbered_name(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(path.name) if path.is_dir() else None
    return int(match.group(1)) if match is not None else -1


def find_resume_candidate(output_root: Path, plan: VideoPlan) -> ResumeCandidate | None:
    """最新の全長原本が再利用可能かを既存物の読取りだけで判定する。"""

    parent = _attempt_root(output_root, plan.target.target_id)
    if not parent.is_dir():
        return None
    retries = sorted(
        (path for path in parent.iterdir() if _numbered_name(path, RETRY_PATTERN) >= 0),
        key=lambda path: _numbered_name(path, RETRY_PATTERN), reverse=True,
    )
    for root in retries:
        index = _numbered_name(root, RETRY_PATTERN)
        retry = _retry_paths(root, plan.target.target_id, index)
        run = load_full_run_reference(retry, plan)
        if run is None:
            continue
        state = latest_validation_state(retry, run)
        if state == "failed":
            return None
        return ResumeCandidate(run, state == "passed")
    return None


def load_full_run_reference(retry: RetryPaths, plan: VideoPlan) -> RunReference | None:
    """runner結果・完了印・manifestの全長一致を確認する。"""

    result = _read_json_optional(retry.run_result)
    if result is None or not _result_covers_full_video(result, plan.metadata.frame_count):
        return None
    run_text = result.get("run_dir")
    if not isinstance(run_text, str) or not run_text:
        return None
    run_dir = Path(run_text).resolve()
    if not run_dir.is_relative_to(retry.event_runs.resolve()):
        return None
    if any(not (run_dir / name).is_file() for name in RUN_MARKERS):
        return None
    manifest = _read_json_optional(run_dir / "manifest.json")
    if manifest is None or not _manifest_covers_full_video(manifest, plan):
        return None
    return RunReference(run_dir, retry)


def _result_covers_full_video(result: Mapping[str, object], frame_count: int) -> bool:
    requested = result.get("requested_end_frame_exclusive")
    processed = result.get("processed_end_frame_exclusive")
    return _exact_int(requested, frame_count) and _exact_int(processed, frame_count)


def _manifest_covers_full_video(manifest: Mapping[str, object], plan: VideoPlan) -> bool:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        return False
    return all((
        source.get("source_video_id") == plan.video_path.stem,
        _exact_int(source.get("frame_count"), plan.metadata.frame_count),
        _exact_int(source.get("processing_start_frame"), 0),
        _exact_int(source.get("processing_end_frame_exclusive"), plan.metadata.frame_count),
    ))


def latest_validation_state(retry: RetryPaths, run: RunReference) -> str:
    """最新roundをpassed/failed/incompleteへ分類する。"""

    if not retry.validation_root.is_dir():
        return "incomplete"
    rounds = sorted(
        (path for path in retry.validation_root.iterdir()
         if _numbered_name(path, ROUND_PATTERN) >= 0),
        key=lambda path: _numbered_name(path, ROUND_PATTERN), reverse=True,
    )
    if not rounds:
        return "incomplete"
    summary = _read_json_optional(rounds[0] / "results" / "summary.json")
    if summary is None or summary.get("run_dir") != str(run.run_dir):
        return "incomplete"
    all_pass = summary.get("all_pass")
    if not isinstance(all_pass, bool):
        return "incomplete"
    return "passed" if all_pass else "failed"


def build_runner_command(plan: VideoPlan, retry: RetryPaths) -> tuple[str, ...]:
    """既存単体runnerを開始0・全長超過max_secで呼ぶ。"""

    return (
        sys.executable, "-u", "-m", "scripts.run_event_snapshot_pilot_v1",
        "--video", str(plan.video_path),
        "--work-npz", str(retry.work_npz),
        "--log", str(retry.collection_log),
        "--output-root", str(retry.event_runs),
        "--attempt-id", retry.attempt_id,
        "--start-sec", "0",
        "--max-sec", repr(plan.max_sec),
        "--score-region-calibration", str(plan.calibration_path),
        "--result-json", str(retry.run_result),
    )


def run_collection(
    plan: VideoPlan, retry: RetryPaths, command_runner: CommandRunner,
) -> RunReference:
    """単体runnerを一回だけ実行し、全長結果以外を拒否する。"""

    command = build_runner_command(plan, retry)
    return_code = command_runner(command, PROJECT_ROOT, retry.root / "logs" / "runner.log")
    if return_code != 0:
        raise RuntimeError(f"単体runnerが終了コード{return_code}で失敗しました")
    run = load_full_run_reference(retry, plan)
    if run is None:
        raise RuntimeError(
            f"processed_end_frame_exclusiveがframe_countと一致する完了runがありません: "
            f"{plan.target.target_id}"
        )
    return run


def build_validation_commands(
    run: RunReference, paths: ValidationPaths,
) -> dict[str, tuple[tuple[str, ...], Path, Path]]:
    """3検証器の出力とログを同一round内で完全分離する。"""

    common = (sys.executable, "-u", "-m")
    return {
        "observation": (
            (*common, "scripts.verify_event_observation_pilot_v1", "--run-dir",
             str(run.run_dir), "--out", str(paths.results / "observation.json")),
            paths.logs / "observation.log", paths.results / "observation.json",
        ),
        "accounting": (
            (*common, "scripts.verify_event_accounting_pilot_v1", "--run-dir",
             str(run.run_dir), "--out", str(paths.results / "accounting.json")),
            paths.logs / "accounting.log", paths.results / "accounting.json",
        ),
        "cross": (
            (*common, "scripts.verify_event_exchange_cross_v1", "--run-dir",
             str(run.run_dir), "--output", str(paths.results / "cross.json")),
            paths.logs / "cross.log", paths.results / "cross.json",
        ),
    }


def run_validations(
    run: RunReference, command_runner: CommandRunner,
) -> ValidationResult:
    """3検証を順次実行し、cross隔離数を失敗条件から分離する。"""

    paths = reserve_validation_round(run.retry)
    commands = build_validation_commands(run, paths)
    codes: dict[str, int] = {}
    reports: dict[str, dict[str, object] | None] = {}
    for name, (command, log_path, result_path) in commands.items():
        try:
            codes[name] = command_runner(command, PROJECT_ROOT, log_path)
        except Exception as exc:  # subprocess起動事故も他検証の集計を止めない
            codes[name] = -1
            _record_command_exception(log_path, exc)
        reports[name] = _read_json_optional(result_path)
    summary = evaluate_validation_reports(run, codes, reports, paths.round_index)
    _write_json_exclusive(paths.results / "summary.json", summary)
    return ValidationResult(bool(summary["all_pass"]), summary, paths.root)


def _record_command_exception(log_path: Path, exc: Exception) -> None:
    """既存ログを変更せず子process起動事故を別の排他記録へ残す。"""

    error_path = log_path.with_name(f"{log_path.stem}-launch-error.json")
    _write_json_exclusive(error_path, {"error_type": type(exc).__name__, "error": str(exc)})


def evaluate_validation_reports(
    run: RunReference, return_codes: Mapping[str, int],
    reports: Mapping[str, Mapping[str, object] | None], round_index: int,
) -> dict[str, object]:
    """既存検証器のPASSとcross会計分類を統合する。"""

    observation = reports.get("observation")
    accounting = reports.get("accounting")
    cross = reports.get("cross")
    expected_run = str(run.run_dir)
    observation_pass = return_codes.get("observation") == 0 and _observation_pass(
        observation, expected_run,
    )
    accounting_pass = return_codes.get("accounting") == 0 and _accounting_pass(
        accounting, expected_run,
    )
    cross_stats = _cross_stats(cross, expected_run)
    cross_pass = return_codes.get("cross") == 0 and cross_stats is not None
    cross_pass = cross_pass and bool(cross_stats["quarantine_contract_pass"])
    all_pass = observation_pass and accounting_pass and cross_pass
    return {
        "format": VALIDATION_FORMAT,
        "run_dir": str(run.run_dir),
        "retry_index": run.retry.retry_index,
        "round_index": round_index,
        "return_codes": dict(return_codes),
        "observation_pass": observation_pass,
        "accounting_pass": accounting_pass,
        "cross_pass": cross_pass,
        "cross": cross_stats,
        "all_pass": all_pass,
    }


def _single_run(
    report: Mapping[str, object] | None, expected_run: str,
) -> Mapping[str, object] | None:
    if report is None or report.get("all_pass") is not True:
        return None
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        return None
    run = runs[0]
    if not isinstance(run, Mapping) or run.get("run_dir") != expected_run:
        return None
    return run


def _observation_pass(
    report: Mapping[str, object] | None, expected_run: str,
) -> bool:
    run = _single_run(report, expected_run)
    return run is not None and run.get("validation_pass") is True


def _accounting_pass(
    report: Mapping[str, object] | None, expected_run: str,
) -> bool:
    return _single_run(report, expected_run) is not None


def _cross_stats(
    report: Mapping[str, object] | None, expected_run: str,
) -> dict[str, object] | None:
    if report is None or report.get("run_dir") != expected_run:
        return None
    if report.get("schema_version") != "event-exchange-cross-validation/v1":
        return None
    physical = report.get("physical_gate_pass")
    balance = report.get("confirmed_balance_gate_pass")
    physical_count = report.get("physical_candidate_count")
    high_count = report.get("high_confidence_candidate_count")
    low_unsupported = report.get("low_confidence_unsupported_count")
    usable = report.get("high_confidence_balance_usable_count")
    quarantined = report.get("high_confidence_balance_quarantined_count")
    high_unsupported = report.get("high_confidence_unsupported_count")
    provisional = report.get("provisional_used_in_confirmed_balance_count")
    allocated = report.get("allocated_supply_amount")
    overuse = report.get("allocated_supply_overuse_count")
    if not isinstance(physical, bool) or not isinstance(balance, bool):
        return None
    counts = (
        physical_count, high_count, low_unsupported, usable, quarantined,
        high_unsupported, provisional, allocated, overuse,
    )
    if any(not _is_nonnegative_int(value) for value in counts):
        return None
    if int(high_count) != int(usable) + int(quarantined):
        return None
    if int(physical_count) != int(high_count) + int(low_unsupported):
        return None
    if int(provisional) != 0 or int(overuse) != 0:
        return None
    quarantine_stats = _quarantine_stats(
        report.get("cross_quarantine"), physical, high_unsupported,
    )
    if quarantine_stats is None:
        return None
    return {
        "physical_gate_pass": physical,
        "confirmed_balance_gate_pass": balance,
        "physical_candidate_count": physical_count,
        "high_confidence_candidate_count": high_count,
        "low_confidence_unsupported_count": low_unsupported,
        "high_confidence_balance_usable_count": usable,
        "high_confidence_balance_quarantined_count": quarantined,
        "high_confidence_unsupported_count": high_unsupported,
        "provisional_used_in_confirmed_balance_count": provisional,
        "allocated_supply_amount": allocated,
        "allocated_supply_overuse_count": overuse,
        **quarantine_stats,
    }


def _quarantine_stats(
    value: object, physical_gate_pass: bool, high_unsupported: object,
) -> dict[str, object] | None:
    """物理完全一致とは独立した行単位隔離契約をfail-closedで読む。"""

    if value is None and physical_gate_pass:
        return {
            "quarantine_contract_pass": True,
            "unsupported_game_count": 0,
        }
    if not isinstance(value, Mapping):
        return None
    integer_fields = (
        "unsupported_high_confidence_count", "unsupported_game_count",
        "unsafe_committed_allocation_count", "usable_uncommitted_allocation_count",
        "low_confidence_allocation_count",
    )
    if any(not _is_nonnegative_int(value.get(name)) for name in integer_fields):
        return None
    indices = value.get("unsupported_game_indices")
    if not isinstance(indices, list) or any(not _is_nonnegative_int(row) for row in indices):
        return None
    safe = (
        value.get("count_consistent") is True
        and int(value["unsafe_committed_allocation_count"]) == 0
        and int(value["usable_uncommitted_allocation_count"]) == 0
        and int(value["low_confidence_allocation_count"]) == 0
        and int(value["unsupported_high_confidence_count"]) == int(high_unsupported)
        and len(set(int(row) for row in indices)) == int(value["unsupported_game_count"])
        and value.get("all_physical_candidates_explained") is physical_gate_pass
    )
    if value.get("contract_pass") is not safe:
        return None
    return {
        "quarantine_contract_pass": safe,
        "unsupported_game_count": value["unsupported_game_count"],
    }


def _execute_command(command: tuple[str, ...], cwd: Path, log_path: Path) -> int:
    """shellを介さず子processを実行し、ログを排他作成する。"""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        result = subprocess.run(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return int(result.returncode)


def process_target(
    plan: VideoPlan, output_root: Path, status: StatusSink,
    command_runner: CommandRunner = _execute_command,
) -> TargetOutcome:
    """一動画を既存run再検証、検証再開、新規retryの順で処理する。"""

    target_id = plan.target.target_id
    status.emit(target_id, "inspection_started", None)
    candidate = find_resume_candidate(output_root, plan)
    if candidate is not None:
        state = "revalidation_started" if candidate.prior_validation_pass else "validation_resumed"
        status.emit(target_id, state, candidate.run.retry.retry_index)
        validation = run_validations(candidate.run, command_runner)
        outcome_state = _resumed_outcome_state(candidate, validation)
        return _finish_outcome(plan, candidate.run, validation, outcome_state, status)
    retry = reserve_retry(output_root, target_id)
    status.emit(target_id, "collection_started", retry.retry_index,
                {"attempt_id": retry.attempt_id})
    run = run_collection(plan, retry, command_runner)
    status.emit(target_id, "validation_started", retry.retry_index,
                {"run_dir": str(run.run_dir)})
    validation = run_validations(run, command_runner)
    outcome_state = "completed" if validation.all_pass else "failed"
    return _finish_outcome(plan, run, validation, outcome_state, status)


def _resumed_outcome_state(
    candidate: ResumeCandidate, validation: ValidationResult,
) -> str:
    if not validation.all_pass:
        return "failed"
    return "revalidated_skip" if candidate.prior_validation_pass else "validation_resumed_completed"


def _finish_outcome(
    plan: VideoPlan, run: RunReference, validation: ValidationResult,
    state: str, status: StatusSink,
) -> TargetOutcome:
    detail = {
        "attempt_id": run.retry.attempt_id,
        "run_dir": str(run.run_dir),
        "validation_round": validation.round_root.name,
    }
    status.emit(plan.target.target_id, state, run.retry.retry_index, detail)
    error = None if validation.all_pass else "3検証器のPASS条件を満たしません"
    return TargetOutcome(
        plan.target.target_id, state, run.retry.retry_index, run.retry.attempt_id,
        str(run.run_dir), validation.summary, error,
    )


def process_target_safely(
    plan: VideoPlan, output_root: Path, status: StatusSink,
    command_runner: CommandRunner = _execute_command,
) -> TargetOutcome:
    """一動画の失敗を他動画から隔離し、次回retryへ残す。"""

    try:
        return process_target(plan, output_root, status, command_runner)
    except Exception as exc:
        retry = _latest_retry_index(output_root, plan.target.target_id)
        status.emit(plan.target.target_id, "failed", retry, {"error": str(exc)})
        return TargetOutcome(
            plan.target.target_id, "failed", retry, None, None, None, str(exc),
        )


def _latest_retry_index(output_root: Path, target_id: str) -> int | None:
    parent = _attempt_root(output_root, target_id)
    if not parent.is_dir():
        return None
    indices = [_numbered_name(path, RETRY_PATTERN) for path in parent.iterdir()]
    valid = [index for index in indices if index >= 0]
    return max(valid) if valid else None


def run_parallel_targets(
    plans: Sequence[VideoPlan], output_root: Path, workers: int,
    command_runner: CommandRunner = _execute_command,
) -> tuple[TargetOutcome, ...]:
    """対象動画を最大3並列で独立処理する。"""

    if workers < MIN_WORKERS or workers > MAX_WORKERS:
        raise ValueError(f"workersは{MIN_WORKERS}〜{MAX_WORKERS}に限定します")
    with exclusive_run_lock(output_root):
        completed = _run_locked_targets(plans, output_root, workers, command_runner)
    return tuple(completed[plan.target.target_id] for plan in plans)


def _run_locked_targets(
    plans: Sequence[VideoPlan], output_root: Path, workers: int,
    command_runner: CommandRunner,
) -> dict[str, TargetOutcome]:
    """一つの出力先を占有した状態で対象を並列処理する。"""
    status, completed = StatusSink(output_root), {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_target_safely, plan, output_root, status, command_runner,
            ): plan.target.target_id for plan in plans
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    return completed


@contextmanager
def exclusive_run_lock(output_root: Path) -> Iterator[None]:
    """同じ出力先の二重起動をOSロックで拒否し、異常終了時も自動解放する。"""
    output_root.resolve().mkdir(parents=True, exist_ok=True)
    handle = (output_root.resolve() / ".event-pilot-48-v1.lock").open("a+b")
    try:
        _acquire_file_lock(handle)
        yield
    finally:
        _release_file_lock(handle)
        handle.close()


def _acquire_file_lock(handle: BinaryIO) -> None:
    try:
        if os.name == "nt":
            _lock_windows_file(handle)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("同じ出力先の48本処理が既に実行中です") from exc


def _lock_windows_file(handle: BinaryIO) -> None:
    import msvcrt
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)


def _release_file_lock(handle: BinaryIO) -> None:
    if handle.closed:
        return
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def aggregate_outcomes(
    outcomes: Sequence[TargetOutcome], selected_count: int,
) -> dict[str, object]:
    """動画成否とcross両ゲート・利用可・隔離を全件集計する。"""

    cross_rows = [
        outcome.validation.get("cross")
        for outcome in outcomes if outcome.validation is not None
    ]
    valid_cross = [row for row in cross_rows if isinstance(row, Mapping)]
    completed_states = {"completed", "revalidated_skip", "validation_resumed_completed"}
    completed_count = sum(outcome.state in completed_states for outcome in outcomes)
    return {
        "format": SUMMARY_FORMAT,
        "selected_target_count": selected_count,
        "completed_target_count": completed_count,
        "failed_target_count": len(outcomes) - completed_count,
        "all_pass": completed_count == selected_count,
        "cross_totals": _aggregate_cross_rows(valid_cross),
        "outcomes": [_outcome_value(outcome) for outcome in outcomes],
    }


def _aggregate_cross_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """動画別cross要約を母数を落とさず合算する。"""
    return {
        "reported_count": len(rows),
        "physical_gate_pass_count": sum(row["physical_gate_pass"] is True for row in rows),
        "physical_gate_fail_count": sum(row["physical_gate_pass"] is False for row in rows),
        "confirmed_balance_gate_pass_count": sum(
            row["confirmed_balance_gate_pass"] is True for row in rows
        ),
        "confirmed_balance_gate_fail_count": sum(
            row["confirmed_balance_gate_pass"] is False for row in rows
        ),
        "high_confidence_balance_usable_count": sum(
            int(row["high_confidence_balance_usable_count"]) for row in rows
        ),
        "high_confidence_balance_quarantined_count": sum(
            int(row["high_confidence_balance_quarantined_count"]) for row in rows
        ),
        "physical_candidate_count": sum(
            int(row["physical_candidate_count"]) for row in rows
        ),
        "low_confidence_unsupported_count": sum(
            int(row["low_confidence_unsupported_count"]) for row in rows
        ),
        "allocated_supply_amount": sum(
            int(row["allocated_supply_amount"]) for row in rows
        ),
        "provisional_used_in_confirmed_balance_count": sum(
            int(row["provisional_used_in_confirmed_balance_count"]) for row in rows
        ),
        "allocated_supply_overuse_count": sum(
            int(row["allocated_supply_overuse_count"]) for row in rows
        ),
    }


def _outcome_value(outcome: TargetOutcome) -> dict[str, object]:
    return {
        "target_id": outcome.target_id,
        "state": outcome.state,
        "retry_index": outcome.retry_index,
        "attempt_id": outcome.attempt_id,
        "run_dir": outcome.run_dir,
        "validation": outcome.validation,
        "error": outcome.error,
    }


def build_dry_run_summary(
    plans: Sequence[VideoPlan], all_pilot_count: int,
) -> dict[str, object]:
    """書込みもsubprocess実行もしない事前検査結果を返す。"""

    return {
        "format": SUMMARY_FORMAT,
        "dry_run": True,
        "manifest_all_count": EXPECTED_ALL_COUNT,
        "manifest_pilot_count": all_pilot_count,
        "manifest_reserve_count": EXPECTED_RESERVE_COUNT,
        "selected_target_count": len(plans),
        "targets": [
            {
                "target_id": plan.target.target_id,
                "video": str(plan.video_path),
                "frame_count": plan.metadata.frame_count,
                "fps": plan.metadata.fps,
                "max_sec": plan.max_sec,
                "calibration": str(plan.calibration_path),
            }
            for plan in plans
        ],
    }


def execute(args: argparse.Namespace) -> dict[str, object]:
    """全件preflight後、dry-runまたはpilot限定実行を行う。"""

    pilot, _reserve = load_fixed_manifest()
    calibrations = load_calibration_map(CALIBRATION_DIR, pilot)
    all_plans = build_video_plans(PROJECT_ROOT, pilot, calibrations)
    plans = select_pilot_plans(all_plans, args.targets)
    if args.dry_run:
        return build_dry_run_summary(plans, len(all_plans))
    outcomes = run_parallel_targets(plans, args.output_root, args.workers)
    summary = aggregate_outcomes(outcomes, len(plans))
    summary_path = write_summary_exclusive(args.output_root, summary)
    return {**summary, "summary_path": str(summary_path)}


def write_summary_exclusive(output_root: Path, summary: Mapping[str, object]) -> Path:
    """実行要約をUUID付き新規JSONへ保存する。"""

    directory = output_root.resolve() / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"summary-{time.time_ns()}-{uuid.uuid4().hex}.json"
    _write_json_exclusive(path, summary)
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return value


def _read_json_optional(path: Path) -> dict[str, object] | None:
    try:
        return _read_json_object(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return payload.encode("utf-8")


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    _write_bytes_exclusive(path, _canonical_bytes(value))


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exact_int(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """reserveを選ぶ入口を持たないCLI引数を読む。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true",
                        help="人工テスト用。preflightのみで書込み・子process実行なし")
    parser.add_argument("--targets", action="append",
                        help="人工テスト用。固定pilot内のtargetだけへ縮小（reserve指定不可）")
    args = parser.parse_args(argv)
    if args.workers < MIN_WORKERS or args.workers > MAX_WORKERS:
        parser.error(f"workersは{MIN_WORKERS}〜{MAX_WORKERS}に限定します")
    return args


def main() -> int:
    """CLIエントリポイント。"""

    args = parse_args()
    summary = execute(args)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if args.dry_run or summary.get("all_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
