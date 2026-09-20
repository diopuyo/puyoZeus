"""未処理の保持側動画を1本ずつ解析し、暫定30本とは分離して保存する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from scripts.run_event_pilot_48_v1 import (
    ManifestTarget,
    VideoMetadata,
    full_length_max_sec,
    load_fixed_manifest,
    probe_video,
)
from src.event_accounting_pilot_validation_v1 import (
    validate_accounting_pilot_loaded_run,
)
from src.event_exchange_cross_validation_v1 import (
    audit_exchange_cross_quarantine_v1,
    cross_validate_exchange_events_v1,
)
from src.event_observation_pilot_validation_v1 import (
    validate_observation_pilot_loaded_run,
)
from src.event_pilot_analysis_v1 import load_prevalidated_run_events
from src.event_run_v1 import validate_completed_run
from src.event_snapshot_export_v1 import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data/verify/event_reserve_background_2026-08-31"
CALIBRATION_ROOT = PROJECT_ROOT / "data/verify/score_region_calibration_reserve_2026-08-31"
LOG_ROOT = PROJECT_ROOT / "logs/event_reserve_background_2026-08-31"
EXCLUDED_TARGETS = frozenset({"c109"})
CALIBRATION_START_SEC = 180.0
CALIBRATION_WINDOW_SEC = 600.0
CALIBRATION_SAMPLE_COUNT = 34
MONITOR_INTERVAL_SEC = 20 * 60
POLL_INTERVAL_SEC = 60
QUEUE_FORMAT = "event-reserve-background-queue/v1"
CAMPAIGN_CLAIM_FORMAT = "event-reserve-background-campaign-claim/v1"
CAMPAIGN_SUMMARY_FORMAT = "event-reserve-background-campaign-summary/v1"
CAMPAIGN_COMPLETE_FORMAT = "event-reserve-background-campaign-complete/v1"
ATTEMPT_NAMESPACE = "event-reserve-background-v1"
ATTEMPT_DATE = "20260831"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data/frames"
C96_SOURCE_STEM = "video_c96"
C96_CAMPAIGN_CLAIM_NAME = "event_reserve_c96_fixed_campaign_v1.claim"
C96_CAMPAIGN_LOCK_NAME = "event_reserve_c96_fixed_campaign_v1.lock"
C96_CAMPAIGN_ATTEMPTS_DIR = "attempts"
C96_RECOVERY_TARGET_IDS = ("c96s1", "c96s2", "c96s3")
C96_CAMPAIGN_KIND = "c96-clip-recovery"
C36_C60_CAMPAIGN_CLAIM_NAME = "event_reserve_c36_c60_full_campaign_v1.claim"
C36_C60_CAMPAIGN_LOCK_NAME = "event_reserve_c36_c60_full_campaign_v1.lock"
C36_C60_RECOVERY_TARGET_IDS = ("c36", "c60")
C36_C60_CAMPAIGN_KIND = "c36-c60-full-recollection"
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_C96_CAMPAIGN_LOCK_HANDLE: BinaryIO | None = None
_C36_C60_CAMPAIGN_LOCK_HANDLE: BinaryIO | None = None


@dataclass(frozen=True, slots=True)
class ExecutionLayout:
    """一回のqueueに閉じた全書込みrootとattempt名前空間。"""

    output_root: Path
    calibration_root: Path
    log_root: Path
    claim_root: Path
    attempt_namespace: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class TargetPaths:
    """一対象の上書き禁止成果物位置。"""

    target_id: str
    video: Path
    calibration: Path
    calibration_log: Path
    work_npz: Path
    collection_log: Path
    launcher_log: Path
    result_json: Path
    runs: Path
    audit_json: Path
    attempt_id: str
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    """versioned campaignの一対象について監査まで終えた結果。"""

    target_id: str
    state: str
    contract_pass: bool
    error: str | None = None
    audit_sha256: str | None = None
    run_dir: str | None = None
    run_artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """物理入力を共有する復旧campaignの固定契約。"""

    kind: str
    ordered_target_ids: tuple[str, ...]
    source_stems: tuple[str, ...]
    claim_name: str
    lock_name: str
    full_length: bool


C96_CAMPAIGN = CampaignSpec(
    C96_CAMPAIGN_KIND, C96_RECOVERY_TARGET_IDS, (C96_SOURCE_STEM,),
    C96_CAMPAIGN_CLAIM_NAME, C96_CAMPAIGN_LOCK_NAME, False,
)
C36_C60_CAMPAIGN = CampaignSpec(
    C36_C60_CAMPAIGN_KIND, C36_C60_RECOVERY_TARGET_IDS,
    ("video_c36", "video_c60"), C36_C60_CAMPAIGN_CLAIM_NAME,
    C36_C60_CAMPAIGN_LOCK_NAME, True,
)
CAMPAIGN_SPECS = (C96_CAMPAIGN, C36_C60_CAMPAIGN)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument(
        "--artifact-version",
        help="明示targetを既存成果物と分離した新規namespaceへ保存する版ID",
    )
    args = parser.parse_args(argv)
    if args.artifact_version is not None and not args.target:
        parser.error("--artifact-versionには1件以上の--targetが必要です")
    return args


def reserve_targets(requested: Sequence[str] = ()) -> tuple[ManifestTarget, ...]:
    """固定台帳の保持側から、レビュー候補c109を除いた順序を返す。"""

    _pilot, reserve = load_fixed_manifest()
    candidates = tuple(row for row in reserve if row.target_id not in EXCLUDED_TARGETS)
    if not requested:
        return candidates
    wanted = tuple(dict.fromkeys(requested))
    known = {row.target_id for row in candidates}
    if len(wanted) != len(requested) or set(wanted) - known:
        raise ValueError("targetはc109以外の保持側IDを重複なく指定してください")
    selected = set(wanted)
    return tuple(row for row in candidates if row.target_id in selected)


def execution_layout(version: str | None = None) -> ExecutionLayout:
    """省略時は従来root、明示版は全成果物を新規rootへ束ねる。"""

    if version is None:
        return ExecutionLayout(
            OUTPUT_ROOT, CALIBRATION_ROOT, LOG_ROOT,
            OUTPUT_ROOT / "active-queue-claim", ATTEMPT_NAMESPACE,
        )
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("artifact-versionは安全なASCII IDにしてください")
    suffix = f"__version={version}"
    layout = ExecutionLayout(
        OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + suffix),
        CALIBRATION_ROOT.with_name(CALIBRATION_ROOT.name + suffix),
        LOG_ROOT.with_name(LOG_ROOT.name + suffix),
        OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + suffix) / "active-queue-claim",
        f"{ATTEMPT_NAMESPACE}-{version}", version,
    )
    _validate_new_layout(layout)
    return layout


def _validate_new_layout(layout: ExecutionLayout) -> None:
    """復旧版rootが既存成果物・raw動画・相互に交差しないことを強制する。"""

    write_roots = (layout.output_root, layout.calibration_root, layout.log_root)
    forbidden = (OUTPUT_ROOT, CALIBRATION_ROOT, LOG_ROOT, RAW_VIDEO_ROOT)
    if any(_paths_overlap(left, right) for left in write_roots for right in forbidden):
        raise ValueError("版付きrootは既存rootおよびraw動画rootと完全分離してください")
    if any(_paths_overlap(left, right) for index, left in enumerate(write_roots)
           for right in write_roots[index + 1:]):
        raise ValueError("版付きoutput/calibration/log rootは相互分離してください")
    candidates = (*write_roots, layout.claim_root)
    if any(path.exists() or path.is_symlink() for path in candidates):
        raise FileExistsError("版付き成果物rootまたはclaim pathが既に存在します")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved, right_resolved = left.resolve(), right.resolve()
    return bool(
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _target_interval_value(target: ManifestTarget) -> dict[str, Any]:
    """target ID・元映像・意味上の秒区間を正規化する。"""

    start, end = _clip_bounds(target)
    return {
        "target_id": target.target_id,
        "source_video_path": target.source_video_path.as_posix(),
        "clip_start_sec": start,
        "clip_end_sec": end,
    }


def _target_interval_sha256(targets: Sequence[ManifestTarget]) -> str:
    payload = json.dumps(
        [_target_interval_value(target) for target in targets],
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_video_sha256(target: ManifestTarget) -> str:
    """campaign開始時点の物理入力SHA-256を返す。"""

    return file_sha256((PROJECT_ROOT / target.source_video_path).resolve())


def _source_contract(targets: Sequence[ManifestTarget]) -> tuple[list[dict[str, str]], str]:
    """同一物理入力を重複排除した順序付きsource契約を作る。"""

    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for target in targets:
        source_path = target.source_video_path.as_posix()
        if source_path in seen:
            continue
        seen.add(source_path)
        sources.append({
            "source_video_path": source_path,
            "source_video_sha256": _source_video_sha256(target),
        })
    payload = json.dumps(
        sources, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sources, hashlib.sha256(payload).hexdigest()


def _fixed_campaign_claim_path() -> Path:
    """artifact versionに依存しないc96物理入力の固定claimを返す。"""

    return OUTPUT_ROOT.parent / C96_CAMPAIGN_CLAIM_NAME


def _fixed_campaign_lock_path() -> Path:
    """c96物理入力の同時処理だけを防ぐOS lock fileを返す。"""

    return OUTPUT_ROOT.parent / C96_CAMPAIGN_LOCK_NAME


def _campaign_claim_path(spec: CampaignSpec) -> Path:
    return OUTPUT_ROOT.parent / spec.claim_name


def _campaign_lock_path(spec: CampaignSpec) -> Path:
    return OUTPUT_ROOT.parent / spec.lock_name


def _uses_c96_source(targets: Sequence[ManifestTarget]) -> bool:
    return any(target.source_video_path.stem == C96_SOURCE_STEM for target in targets)


def _campaign_spec_for_ids(target_ids: Sequence[str]) -> CampaignSpec | None:
    ids = tuple(target_ids)
    return next((spec for spec in CAMPAIGN_SPECS if ids == spec.ordered_target_ids), None)


def _campaign_specs_for_targets(targets: Sequence[ManifestTarget]) -> tuple[CampaignSpec, ...]:
    stems = {target.source_video_path.stem for target in targets}
    return tuple(spec for spec in CAMPAIGN_SPECS if stems.intersection(spec.source_stems))


def _campaign_targets(
    spec: CampaignSpec, targets: Sequence[ManifestTarget],
) -> tuple[ManifestTarget, ...]:
    stems = set(spec.source_stems)
    return tuple(target for target in targets if target.source_video_path.stem in stems)


def _validate_versioned_campaign_request(
    requested: Sequence[str], version: str | None,
) -> CampaignSpec | None:
    """版付きCLIは二つの固定campaignのexact ordered setだけを許す。"""

    if version is None:
        return None
    spec = _campaign_spec_for_ids(requested)
    if spec is None:
        allowed = " または ".join(",".join(item.ordered_target_ids) for item in CAMPAIGN_SPECS)
        raise ValueError(f"artifact-versionのtargetは{allowed}をこの順で全件指定してください")
    return spec


def _validate_versioned_c96_request(
    requested: Sequence[str], version: str | None,
) -> None:
    """旧内部API名を保った版付きcampaign入口検査。"""

    _validate_versioned_campaign_request(requested, version)


def _validate_campaign_target_contract(
    spec: CampaignSpec, targets: Sequence[ManifestTarget],
) -> None:
    """IDだけでなく物理sourceと全長/clip意味も固定する。"""

    if tuple(target.target_id for target in targets) != spec.ordered_target_ids:
        raise ValueError("artifact-versionはexact ordered complete setが必要です")
    expected_stems = (
        spec.source_stems * len(targets)
        if len(spec.source_stems) == 1 else spec.source_stems
    )
    if tuple(target.source_video_path.stem for target in targets) != expected_stems:
        raise ValueError(f"{spec.kind}のsource動画が固定契約と一致しません")
    bounds = tuple(_clip_bounds(target) for target in targets)
    if spec.full_length and any(bound != (None, None) for bound in bounds):
        raise ValueError(f"{spec.kind}はmanifestのclip指定なし・全長処理が必須です")
    if not spec.full_length and any(start is None or end is None for start, end in bounds):
        raise ValueError(f"{spec.kind}はmanifestの全clip境界が必須です")


def _validate_versioned_campaign_targets(
    targets: Sequence[ManifestTarget], layout: ExecutionLayout,
) -> CampaignSpec | None:
    """正規化後も版付きcampaignの完全性を成果物作成前に再検証する。"""

    if layout.version is None:
        return None
    spec = _campaign_spec_for_ids(target.target_id for target in targets)
    if spec is None:
        raise ValueError("artifact-versionはexact ordered complete setが必要です")
    _validate_campaign_target_contract(spec, targets)
    return spec


def _validate_versioned_c96_targets(
    targets: Sequence[ManifestTarget], layout: ExecutionLayout,
) -> None:
    """旧内部API名を保った版付きcampaign成果物前検査。"""

    _validate_versioned_campaign_targets(targets, layout)


def _lock_file_nonblocking(handle: BinaryIO) -> None:
    """platform標準のadvisory lockをprocess終了まで非待機で保持する。"""

    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    """明示終了時にadvisory lockを解放する。異常終了時はOSが解放する。"""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _campaign_lock_handle(spec: CampaignSpec) -> BinaryIO | None:
    if spec is C96_CAMPAIGN:
        return _C96_CAMPAIGN_LOCK_HANDLE
    return _C36_C60_CAMPAIGN_LOCK_HANDLE


def _set_campaign_lock_handle(spec: CampaignSpec, handle: BinaryIO | None) -> None:
    global _C96_CAMPAIGN_LOCK_HANDLE, _C36_C60_CAMPAIGN_LOCK_HANDLE
    if spec is C96_CAMPAIGN:
        _C96_CAMPAIGN_LOCK_HANDLE = handle
    else:
        _C36_C60_CAMPAIGN_LOCK_HANDLE = handle


def _acquire_campaign_lock(spec: CampaignSpec) -> None:
    """物理入力campaignのprocess-lifetime単一writer lockを取る。"""

    if _campaign_lock_handle(spec) is not None:
        raise FileExistsError(f"{spec.kind}固定campaignは別実行が処理中です")
    path = _campaign_lock_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"{spec.kind}固定campaign lockにsymlinkは使用できません")
    handle = path.open("a+b")
    try:
        _lock_file_nonblocking(handle)
    except OSError as error:
        handle.close()
        raise FileExistsError(f"{spec.kind}固定campaignは別実行が処理中です") from error
    _set_campaign_lock_handle(spec, handle)


def _acquire_c96_campaign_lock() -> None:
    """旧内部API名を保ったc96 process-lifetime lock取得。"""

    _acquire_campaign_lock(C96_CAMPAIGN)


def _acquire_c36_c60_campaign_lock() -> None:
    """c36/c60全長campaignのprocess-lifetime lock取得。"""

    _acquire_campaign_lock(C36_C60_CAMPAIGN)


def _release_campaign_lock(spec: CampaignSpec) -> None:
    """自processのOS lockだけを解放し、append-only claimは残す。"""

    handle = _campaign_lock_handle(spec)
    if handle is None:
        return
    _set_campaign_lock_handle(spec, None)
    try:
        _unlock_file(handle)
    finally:
        handle.close()


def _release_c96_campaign_lock() -> None:
    """旧内部API名を保ったc96 lock解放。"""

    _release_campaign_lock(C96_CAMPAIGN)


def _release_c36_c60_campaign_lock() -> None:
    """c36/c60全長campaign lock解放。"""

    _release_campaign_lock(C36_C60_CAMPAIGN)


def _release_campaign_locks(specs: Sequence[CampaignSpec]) -> None:
    for spec in reversed(specs):
        _release_campaign_lock(spec)


def _reject_claimed_c96(targets: Sequence[ManifestTarget]) -> None:
    """互換名を保ちつつ、永久claimでなく実行中lockだけを拒否する。"""

    acquired: list[CampaignSpec] = []
    try:
        for spec in _campaign_specs_for_targets(targets):
            _acquire_campaign_lock(spec)
            acquired.append(spec)
    except Exception:
        _release_campaign_locks(acquired)
        raise


def target_paths(
    target: ManifestTarget, layout: ExecutionLayout | None = None,
) -> TargetPaths:
    """同一元動画のclip同士も衝突しないtarget別保存位置を返す。"""

    resolved_layout = layout or execution_layout()
    target_root = resolved_layout.output_root / "targets" / f"target={target.target_id}"
    video = (PROJECT_ROOT / target.source_video_path).resolve()
    clip_start, clip_end = _clip_bounds(target)
    artifact_id = f"video_{target.target_id}"
    return TargetPaths(
        target.target_id, video, resolved_layout.calibration_root / f"{artifact_id}.json",
        resolved_layout.log_root / f"{artifact_id}_calibration.log",
        target_root / "work" / f"{artifact_id}.npz",
        resolved_layout.log_root / f"{artifact_id}_collection.log",
        resolved_layout.log_root / f"{artifact_id}_launcher.log",
        resolved_layout.log_root / f"{artifact_id}_result.json",
        target_root / "runs", target_root / "AUDIT.json",
        f"{resolved_layout.attempt_namespace}-{target.target_id}-{ATTEMPT_DATE}",
        clip_start, clip_end, resolved_layout.output_root,
    )


def _clip_bounds(target: ManifestTarget) -> tuple[float | None, float | None]:
    """manifestのclip境界を両端指定として厳密に読む。"""

    if not target.clip_start_sec and not target.clip_end_sec:
        return None, None
    if not target.clip_start_sec or not target.clip_end_sec:
        raise ValueError(f"clip境界は両端必須です: {target.target_id}")
    try:
        start, end = float(target.clip_start_sec), float(target.clip_end_sec)
    except ValueError as error:
        raise ValueError(f"clip境界が数値ではありません: {target.target_id}") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"clip境界が不正です: {target.target_id}")
    return start, end


def _prepare_claim(
    layout: ExecutionLayout | None = None,
    targets: Sequence[ManifestTarget] = (),
) -> None:
    resolved_layout = layout or execution_layout()
    campaign = _validate_versioned_campaign_targets(targets, resolved_layout)
    specs = _campaign_specs_for_targets(targets)
    _reject_claimed_c96(targets)
    try:
        contract: dict[str, Any] | None = None
        if campaign is not None:
            contract = _campaign_contract(campaign, targets)
            _append_campaign_attempt(resolved_layout, campaign, targets, contract)
        elif _uses_c96_source(targets):
            _append_legacy_c96_attempt(resolved_layout, targets)
        claim = resolved_layout.claim_root
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.mkdir(exist_ok=False)
        value: dict[str, Any] = {
            "format": QUEUE_FORMAT, "pid": os.getpid(),
            "started_unix_ns": time.time_ns(),
        }
        if resolved_layout.version is not None:
            value["artifact_version"] = resolved_layout.version
        if contract is not None:
            value.update(contract)
        elif targets:
            value["ordered_target_ids"] = [target.target_id for target in targets]
            value["target_interval_sha256"] = _target_interval_sha256(targets)
        _write_json_exclusive(claim / "CLAIM.json", value)
    except Exception:
        _release_campaign_locks(specs)
        raise


def _campaign_contract(
    spec: CampaignSpec, targets: Sequence[ManifestTarget],
) -> dict[str, Any]:
    """CLAIM/SUMMARY/COMPLETEで共有する不変campaign契約を返す。"""

    sources, source_hash = _source_contract(targets)
    return {
        "campaign_kind": spec.kind,
        "ordered_target_ids": [target.target_id for target in targets],
        "target_intervals": [_target_interval_value(target) for target in targets],
        "target_interval_sha256": _target_interval_sha256(targets),
        "sources": sources,
        "source_contract_sha256": source_hash,
    }


def _prepare_fixed_campaign_claim(
    layout: ExecutionLayout, targets: Sequence[ManifestTarget],
) -> None:
    """旧内部API名を保ち、該当campaignのattempt履歴を追記する。"""

    if layout.version is None:
        if _uses_c96_source(targets):
            _append_legacy_c96_attempt(layout, targets)
        return
    spec = _validate_versioned_campaign_targets(targets, layout)
    if spec is None:
        raise ValueError("versioned campaignの種別を解決できません")
    _append_campaign_attempt(layout, spec, targets)


def _append_legacy_c96_attempt(
    layout: ExecutionLayout, targets: Sequence[ManifestTarget],
) -> None:
    """default c96の既存claim形式をsource読取りなしで維持する。"""

    attempt_path = _campaign_attempt_path(C96_CAMPAIGN)
    _write_json_exclusive(attempt_path, {
        "format": CAMPAIGN_CLAIM_FORMAT,
        "pid": os.getpid(),
        "claimed_unix_ns": time.time_ns(),
        "artifact_version": layout.version,
        "output_root": str(layout.output_root.resolve()),
        "ordered_target_ids": [target.target_id for target in targets],
        "target_interval_sha256": _target_interval_sha256(targets),
    })


def _campaign_attempt_path(spec: CampaignSpec) -> Path:
    """固定claimを変更せず、新しいattempt専用pathを確保する。"""

    claim = _campaign_claim_path(spec)
    claim.parent.mkdir(parents=True, exist_ok=True)
    if claim.is_symlink():
        raise FileExistsError(f"{spec.kind}固定campaign claimにsymlinkは使用できません")
    claim.mkdir(exist_ok=True)
    attempts = claim / C96_CAMPAIGN_ATTEMPTS_DIR
    if attempts.is_symlink():
        raise FileExistsError(f"{spec.kind}固定campaign attemptsにsymlinkは使用できません")
    return attempts / f"{time.time_ns()}-{uuid.uuid4().hex}.json"


def _append_campaign_attempt(
    layout: ExecutionLayout, spec: CampaignSpec, targets: Sequence[ManifestTarget],
    contract: Mapping[str, Any] | None = None,
) -> None:
    """物理入力campaignの試行を固定namespaceへappend-only追記する。"""

    if layout.version is not None:
        selected = _validate_versioned_campaign_targets(targets, layout)
        if selected is not spec:
            raise ValueError("artifact-versionのcampaign種別が固定契約と一致しません")
    resolved_contract = dict(contract or _campaign_contract(spec, targets))
    attempt_path = _campaign_attempt_path(spec)
    value = {
        "format": CAMPAIGN_CLAIM_FORMAT,
        "pid": os.getpid(),
        "claimed_unix_ns": time.time_ns(),
        "artifact_version": layout.version,
        "output_root": str(layout.output_root.resolve()),
    }
    value.update(resolved_contract)
    _write_json_exclusive(attempt_path, value)


def _emit_status(
    target_id: str, state: str, detail: Mapping[str, Any] | None = None,
    output_root: Path | None = None,
) -> None:
    root = (output_root or OUTPUT_ROOT) / "status" / f"target={target_id}"
    root.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(root / f"{time.time_ns()}-{uuid.uuid4().hex}.json", {
        "format": QUEUE_FORMAT, "recorded_unix_ns": time.time_ns(),
        "target_id": target_id, "state": state, "detail": dict(detail or {}),
    })


def _emit_target_status(
    paths: TargetPaths, state: str, detail: Mapping[str, Any] | None = None,
) -> None:
    _emit_status(paths.target_id, state, detail, paths.output_root)


def _ensure_calibration(
    paths: TargetPaths, metadata: VideoMetadata | None = None,
) -> bool:
    if paths.calibration.is_file():
        return True
    if paths.calibration_log.exists():
        _emit_target_status(paths, "calibration_blocked_existing_log")
        return False
    paths.calibration.parent.mkdir(parents=True, exist_ok=True)
    paths.calibration_log.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata = metadata or probe_video(paths.video)
    start_sec, window_sec = _calibration_window(paths, resolved_metadata)
    command = (
        sys.executable, "-u", "-m", "scripts.calibrate_score_regions",
        "--video", str(paths.video), "--output", str(paths.calibration),
        "--sample-count", str(CALIBRATION_SAMPLE_COUNT),
        "--start-sec", repr(start_sec),
        "--window-sec", repr(window_sec),
    )
    code = _run_logged(command, paths.calibration_log)
    _emit_target_status(paths, "calibration_finished", {"return_code": code})
    return code == 0 and paths.calibration.is_file()


def _run_target(
    paths: TargetPaths, metadata: VideoMetadata | None = None,
) -> int:
    resolved_metadata = metadata or probe_video(paths.video)
    start_sec, max_sec = _collection_window(paths, resolved_metadata)
    start_text = "0" if paths.clip_start_sec is None else repr(start_sec)
    paths.work_npz.parent.mkdir(parents=True, exist_ok=True)
    paths.runs.mkdir(parents=True, exist_ok=True)
    command = (
        sys.executable, "-u", "-m", "scripts.run_event_snapshot_pilot_v1",
        "--video", str(paths.video), "--work-npz", str(paths.work_npz),
        "--log", str(paths.collection_log), "--output-root", str(paths.runs),
        "--attempt-id", paths.attempt_id, "--start-sec", start_text,
        "--max-sec", repr(max_sec),
        "--score-region-calibration", str(paths.calibration),
        "--result-json", str(paths.result_json),
    )
    _emit_target_status(paths, "collection_started")
    code = _run_logged(command, paths.launcher_log)
    _emit_target_status(paths, "collection_finished", {"return_code": code})
    return code


def _collection_window(
    paths: TargetPaths, metadata: VideoMetadata,
) -> tuple[float, float]:
    """targetの全長またはclip処理秒範囲を返す。"""

    if paths.clip_start_sec is None and paths.clip_end_sec is None:
        return 0.0, full_length_max_sec(metadata)
    if paths.clip_start_sec is None or paths.clip_end_sec is None:
        raise ValueError(f"clip境界が片側だけです: {paths.target_id}")
    max_sec = paths.clip_end_sec - paths.clip_start_sec
    _expected_processing_range(paths, metadata)
    return paths.clip_start_sec, max_sec


def _calibration_window(
    paths: TargetPaths, metadata: VideoMetadata,
) -> tuple[float, float]:
    """較正窓を処理対象clipの内側へ制限する。"""

    start_sec, max_sec = _collection_window(paths, metadata)
    if paths.clip_start_sec is None:
        return CALIBRATION_START_SEC, CALIBRATION_WINDOW_SEC
    offset = CALIBRATION_START_SEC if max_sec > CALIBRATION_START_SEC else 0.0
    window_sec = min(CALIBRATION_WINDOW_SEC, max_sec - offset)
    if window_sec <= 0:
        raise ValueError(f"較正可能なclip長がありません: {paths.target_id}")
    return start_sec + offset, window_sec


def _expected_processing_range(
    paths: TargetPaths, metadata: VideoMetadata,
) -> tuple[int, int]:
    """runnerと同じ丸めでtarget固有の期待frame範囲を得る。"""

    if paths.clip_start_sec is None and paths.clip_end_sec is None:
        return 0, metadata.frame_count
    if paths.clip_start_sec is None or paths.clip_end_sec is None:
        raise ValueError(f"clip境界が片側だけです: {paths.target_id}")
    start = int(paths.clip_start_sec * metadata.fps)
    length = int((paths.clip_end_sec - paths.clip_start_sec) * metadata.fps)
    end = start + length
    if start < 0 or length <= 0 or start >= metadata.frame_count or end > metadata.frame_count:
        raise ValueError(f"clipが動画frame範囲外です: {paths.target_id}")
    return start, end


def _wait_for_manual_run(paths: TargetPaths) -> bool:
    """既に開始された対象を奪わず、20分ごとに進捗だけ記録する。"""

    last_report = 0.0
    while not paths.result_json.is_file():
        if not _attempt_active(paths.attempt_id):
            _emit_target_status(paths, "orphaned_partial_output")
            return False
        now = time.monotonic()
        if now - last_report >= MONITOR_INTERVAL_SEC:
            size = paths.collection_log.stat().st_size if paths.collection_log.exists() else 0
            _emit_target_status(paths, "collection_running", {"log_bytes": size})
            print(f"[reserve-background] running {paths.target_id} log_bytes={size}", flush=True)
            last_report = now
        time.sleep(POLL_INTERVAL_SEC)
    return True


def _attempt_active(attempt_id: str) -> bool:
    for command_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if attempt_id in command and "run_event_snapshot_pilot_v1" in command:
            return True
    return False


def _has_partial_collection(paths: TargetPaths) -> bool:
    return any(path.exists() for path in (
        paths.work_npz, paths.collection_log, paths.launcher_log, paths.result_json,
    ))


def _audit_completed(paths: TargetPaths) -> bool:
    metadata = probe_video(paths.video)
    if paths.audit_json.is_file():
        audit = _load_json(paths.audit_json)
        run_dir = _audit_run_dir(paths, audit)
        fresh = _independent_audit(paths, run_dir, metadata)
        if audit.get("contract_pass") is not fresh["contract_pass"]:
            raise ValueError("保存済み監査と独立再監査の合否が一致しません")
        return bool(fresh["contract_pass"])
    result = _load_json(paths.result_json)
    run_dir = _result_run_dir(paths, result, metadata)
    value = _independent_audit(paths, run_dir, metadata)
    _write_json_exclusive(paths.audit_json, value)
    return bool(value["contract_pass"])


def _independent_audit(
    paths: TargetPaths, run_dir: Path, metadata: VideoMetadata,
) -> dict[str, Any]:
    """新規・既存を区別せず全独立検査器から監査値を再計算する。"""

    _validate_run_identity(paths, run_dir, metadata)
    completed = validate_completed_run(run_dir)
    if not completed.valid:
        raise ValueError(f"完成原本が不合格です: {completed.error}")
    manifest, events = load_prevalidated_run_events(run_dir, completed)
    accounting = validate_accounting_pilot_loaded_run(run_dir, manifest, events)
    observation = validate_observation_pilot_loaded_run(run_dir, manifest, events)
    cross_report = cross_validate_exchange_events_v1(events)
    quarantine = audit_exchange_cross_quarantine_v1(cross_report)
    return {
        "format": QUEUE_FORMAT, "target_id": paths.target_id,
        "run_dir": str(run_dir.resolve()), "attempt_id": paths.attempt_id,
        "source_video_path": paths.video.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "source_video_sha256": file_sha256(paths.video),
        "clip_start_sec": paths.clip_start_sec, "clip_end_sec": paths.clip_end_sec,
        "processing_start_frame": manifest["source"]["processing_start_frame"],
        "processing_end_frame_exclusive": manifest["source"]["processing_end_frame_exclusive"],
        "event_count": len(events),
        "conservation_residual_nonzero_count": accounting["conservation_residual_nonzero_count"],
        "conservation_residual_denominator": accounting["conservation_residual_observation_side_count"],
        "observation_validation_pass": observation["validation_pass"],
        "physical_gate_pass": cross_report.physical_gate_pass,
        "unsupported_game_count": quarantine["unsupported_game_count"],
        "contract_pass": bool(
            accounting["conservation_residual_nonzero_count"] == 0
            and observation["validation_pass"] is True
            and quarantine["contract_pass"] is True
        ),
    }


def _audit_run_dir(paths: TargetPaths, audit: Mapping[str, Any]) -> Path:
    """既存監査が当該targetの原本だけを参照することを確認する。"""

    if audit.get("target_id") != paths.target_id:
        raise ValueError("既存監査のtarget_idが一致しません")
    run_text = audit.get("run_dir")
    if not isinstance(run_text, str) or not run_text:
        raise ValueError("既存監査にrun_dirがありません")
    return Path(run_text)


def _result_run_dir(
    paths: TargetPaths, result: Mapping[str, Any], metadata: VideoMetadata,
) -> Path:
    """runner結果の範囲とrun参照をtarget契約へ照合する。"""

    _start, expected_end = _expected_processing_range(paths, metadata)
    for key in ("requested_end_frame_exclusive", "processed_end_frame_exclusive"):
        if not _exact_int(result.get(key), expected_end):
            raise ValueError(f"runner結果の{key}がtarget範囲と一致しません")
    run_text = result.get("run_dir")
    if not isinstance(run_text, str) or not run_text:
        raise ValueError("runner結果にrun_dirがありません")
    return Path(run_text)


def _validate_run_identity(
    paths: TargetPaths, run_dir: Path, metadata: VideoMetadata,
) -> dict[str, Any]:
    """runの所有target・attempt・処理範囲をfail-closedで検証する。"""

    resolved = run_dir.resolve()
    expected_root = paths.runs.resolve()
    if not resolved.is_relative_to(expected_root):
        raise ValueError("run_dirが当該targetのruns配下ではありません")
    attempts = tuple(path.resolve() for path in paths.runs.rglob("attempt=*") if path.is_dir())
    manifests = tuple(paths.runs.rglob("manifest.json"))
    if (len(attempts) != 1 or attempts[0] != resolved
            or len(manifests) != 1 or manifests[0].parent.resolve() != resolved):
        raise ValueError("当該targetのrunが一意ではありません")
    manifest = _load_json(manifests[0])
    if manifest.get("attempt_id") != paths.attempt_id:
        raise ValueError("runのattempt_idがtarget契約と一致しません")
    _validate_manifest_source(paths, manifest, metadata)
    return manifest


def _validate_manifest_source(
    paths: TargetPaths, manifest: Mapping[str, Any], metadata: VideoMetadata,
) -> None:
    """run manifestの入力動画と処理frame範囲を照合する。"""

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("run manifestにsource objectがありません")  # noqa: TRY004
    expected_start, expected_end = _expected_processing_range(paths, metadata)
    frame_fields = {
        "frame_count": metadata.frame_count,
        "processing_start_frame": expected_start,
        "processing_end_frame_exclusive": expected_end,
    }
    invalid_frame = any(
        not _exact_int(source.get(key), value) for key, value in frame_fields.items()
    )
    expected_path = paths.video.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    if (source.get("source_video_id") != paths.video.stem or invalid_frame
            or source.get("source_video_path") != expected_path):
        raise ValueError("run manifestの動画または処理範囲がtarget契約と一致しません")
    if source.get("source_video_sha256") != file_sha256(paths.video):
        raise ValueError("run manifestの元映像SHA-256が現在の入力と一致しません")


def _exact_int(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _run_logged(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            tuple(command), cwd=PROJECT_ROOT, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
    return completed.returncode


def _run_artifact_sha256(run_dir: Path) -> str:
    """完成run配下の全file path/contentを順序付きSHA-256へ固定する。"""

    resolved = run_dir.resolve()
    if not resolved.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError("完成run原本ディレクトリがありません")
    entries: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("完成run原本にsymlinkは使用できません")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("完成run原本に通常file以外を含められません")
        entries.append({
            "relative_path": path.relative_to(resolved).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    if not entries:
        raise ValueError("完成run原本にfileがありません")
    payload = json.dumps(
        entries, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _freeze_versioned_outcome(paths: TargetPaths, state: str) -> TargetOutcome:
    """処理直後のAUDITと完成run全file identityを終端照合用に固定する。"""

    audit_sha256 = file_sha256(paths.audit_json)
    audit = _load_json(paths.audit_json)
    if audit.get("contract_pass") is not True:
        raise ValueError("PASSしていないAUDITを成功outcomeへ固定できません")
    run_dir = _audit_run_dir(paths, audit).resolve()
    if not run_dir.is_relative_to(paths.runs.resolve()):
        raise ValueError("AUDITのrun_dirがtargetのruns配下ではありません")
    run_sha256 = _run_artifact_sha256(run_dir)
    if file_sha256(paths.audit_json) != audit_sha256:
        raise ValueError("固定中にAUDITが変更されました")
    return TargetOutcome(
        paths.target_id, state, True, audit_sha256=audit_sha256,
        run_dir=str(run_dir), run_artifact_sha256=run_sha256,
    )


def _target_outcome(
    paths: TargetPaths, state: str, passed: bool,
    layout: ExecutionLayout | None,
) -> TargetOutcome:
    if not passed or layout is None or layout.version is None:
        return TargetOutcome(paths.target_id, state, passed)
    return _freeze_versioned_outcome(paths, state)


def _process_target(
    target: ManifestTarget, layout: ExecutionLayout | None = None,
) -> TargetOutcome:
    paths = target_paths(target, layout)
    if paths.audit_json.is_file():
        try:
            passed = _audit_completed(paths)
            state = "already_complete" if passed else "existing_audit_failed"
            _emit_target_status(paths, state)
        except Exception as error:  # noqa: BLE001 - 不正な既存監査を受理しない
            _emit_target_status(paths, "existing_audit_invalid", {"error": repr(error)})
            return TargetOutcome(paths.target_id, "existing_audit_invalid", False, repr(error))
        return _target_outcome(paths, state, passed, layout)
    if not paths.video.is_file():
        _emit_target_status(paths, "input_or_calibration_failed")
        return TargetOutcome(paths.target_id, "input_or_calibration_failed", False)
    metadata = probe_video(paths.video)
    if not _ensure_calibration(paths, metadata):
        _emit_target_status(paths, "input_or_calibration_failed")
        return TargetOutcome(paths.target_id, "input_or_calibration_failed", False)
    if _has_partial_collection(paths):
        if not paths.result_json.is_file() and not _wait_for_manual_run(paths):
            return TargetOutcome(paths.target_id, "orphaned_partial_output", False)
    elif _run_target(paths, metadata) != 0:
        return TargetOutcome(paths.target_id, "collection_failed", False)
    try:
        passed = _audit_completed(paths)
        _emit_target_status(paths, "audit_finished", {"contract_pass": passed})
    except Exception as error:  # noqa: BLE001 - 次対象を止めず失敗を証跡化する
        _emit_target_status(paths, "audit_failed", {"error": repr(error)})
        return TargetOutcome(paths.target_id, "audit_failed", False, repr(error))
    state = "audit_passed" if passed else "audit_contract_failed"
    return _target_outcome(paths, state, passed, layout)


def _guard_versioned_target(
    target: ManifestTarget, layout: ExecutionLayout,
) -> TargetOutcome:
    """予期しない例外もcampaign失敗へ閉じ、後続targetを監査可能に保つ。"""

    try:
        return _process_target(target, layout)
    except Exception as error:  # noqa: BLE001 - campaign終端へ全失敗を集約する
        paths = target_paths(target, layout)
        _emit_target_status(paths, "target_failed", {"error": repr(error)})
        return TargetOutcome(paths.target_id, "target_failed", False, repr(error))


def _campaign_all_passed(
    targets: Sequence[ManifestTarget], outcomes: Sequence[TargetOutcome],
) -> bool:
    expected = [target.target_id for target in targets]
    actual = [outcome.target_id for outcome in outcomes]
    return actual == expected and all(outcome.contract_pass for outcome in outcomes)


CAMPAIGN_CONTRACT_KEYS = (
    "campaign_kind", "ordered_target_ids", "target_intervals",
    "target_interval_sha256", "sources", "source_contract_sha256",
)


def _validate_saved_campaign_claim(
    layout: ExecutionLayout, contract: Mapping[str, Any],
) -> None:
    """開始時CLAIMと終了時source/interval契約が同一であることを確認する。"""

    claim_path = layout.claim_root / "CLAIM.json"
    if not claim_path.is_file():
        raise FileNotFoundError("versioned campaignのCLAIM.jsonがありません")
    claim = _load_json(claim_path)
    if claim.get("artifact_version") != layout.version:
        raise ValueError("CLAIMのartifact_versionがcampaignと一致しません")
    if any(claim.get(key) != contract[key] for key in CAMPAIGN_CONTRACT_KEYS):
        raise ValueError("CLAIMのcampaign/source/interval契約が終了時と一致しません")


def _audit_contract_passes(
    target: ManifestTarget, audit: Mapping[str, Any], contract: Mapping[str, Any],
) -> bool:
    """独立監査がtargetのsourceと全長/clip契約を証明するか判定する。"""

    interval = _target_interval_value(target)
    source_hashes = {
        item["source_video_path"]: item["source_video_sha256"]
        for item in contract["sources"]
    }
    return bool(
        audit.get("target_id") == target.target_id
        and audit.get("contract_pass") is True
        and audit.get("source_video_path") == interval["source_video_path"]
        and audit.get("source_video_sha256") == source_hashes[interval["source_video_path"]]
        and audit.get("clip_start_sec") == interval["clip_start_sec"]
        and audit.get("clip_end_sec") == interval["clip_end_sec"]
    )


def _outcome_run_dir(paths: TargetPaths, outcome: TargetOutcome) -> Path:
    """処理時outcomeに固定したrun identityをtarget rootへ照合する。"""

    if not isinstance(outcome.run_dir, str) or not outcome.run_dir:
        raise ValueError("処理時outcomeにrun_dir identityがありません")
    run_dir = Path(outcome.run_dir).resolve()
    if not run_dir.is_relative_to(paths.runs.resolve()):
        raise ValueError("処理時outcomeのrun_dirがtargetのruns配下ではありません")
    for name, value in (
        ("audit_sha256", outcome.audit_sha256),
        ("run_artifact_sha256", outcome.run_artifact_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"処理時outcomeに有効な{name}がありません")
    return run_dir


def _terminal_reaudit(
    paths: TargetPaths, outcome: TargetOutcome,
) -> dict[str, Any]:
    """固定run原本から独立監査を再実行し、処理時identityと完全照合する。"""

    run_dir = _outcome_run_dir(paths, outcome)
    fresh = _independent_audit(paths, run_dir, probe_video(paths.video))
    audit = _load_json(paths.audit_json)
    audit_run_dir = _audit_run_dir(paths, audit).resolve()
    if audit_run_dir != run_dir or audit != fresh:
        raise ValueError("終端再監査が処理時AUDIT/run identityと一致しません")
    if file_sha256(paths.audit_json) != outcome.audit_sha256:
        raise ValueError("処理時以降にAUDIT原本が変更されました")
    if _run_artifact_sha256(run_dir) != outcome.run_artifact_sha256:
        raise ValueError("処理時以降に完成run原本が変更されました")
    return audit


def _campaign_result(
    target: ManifestTarget, outcome: TargetOutcome, layout: ExecutionLayout,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    paths = target_paths(target, layout)
    if not paths.audit_json.is_file():
        raise FileNotFoundError(f"監査原本がありません: {target.target_id}")
    audit = _terminal_reaudit(paths, outcome)
    if not _audit_contract_passes(target, audit, contract):
        raise ValueError(f"監査原本がcampaign PASSを証明しません: {target.target_id}")
    return {
        "target_id": outcome.target_id, "state": outcome.state,
        "contract_pass": True,
        "audit_relative_path": paths.audit_json.relative_to(layout.output_root).as_posix(),
        "audit_sha256": outcome.audit_sha256,
        "run_dir": outcome.run_dir,
        "run_artifact_sha256": outcome.run_artifact_sha256,
    }


def _campaign_results(
    targets: Sequence[ManifestTarget], outcomes: Sequence[TargetOutcome],
    layout: ExecutionLayout, contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """全targetを終端再監査し、一件でも不一致なら結果全体を破棄する。"""

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for target, outcome in zip(targets, outcomes):
        try:
            results.append(_campaign_result(target, outcome, layout, contract))
        except Exception as error:  # noqa: BLE001 - 全targetを再監査して集約する
            errors.append(f"{target.target_id}: {error}")
    if errors:
        raise ValueError("終端独立再監査に失敗しました: " + "; ".join(errors))
    return results


def _campaign_summary(
    targets: Sequence[ManifestTarget], outcomes: Sequence[TargetOutcome],
    layout: ExecutionLayout,
) -> dict[str, Any]:
    """全監査PASS時だけ書けるcampaign終端要約を組み立てる。"""

    if layout.version is None:
        raise ValueError("default queueにはcampaign SUMMARYを書けません")
    spec = _validate_versioned_campaign_targets(targets, layout)
    if spec is None:
        raise ValueError("versioned campaignの種別を解決できません")
    if not _campaign_all_passed(targets, outcomes):
        raise ValueError("全targetの独立監査PASSなしにSUMMARYは作成できません")
    contract = _campaign_contract(spec, targets)
    _validate_saved_campaign_claim(layout, contract)
    results = _campaign_results(targets, outcomes, layout, contract)
    return {
        "format": CAMPAIGN_SUMMARY_FORMAT, "artifact_version": layout.version,
        **contract, "target_count": len(targets), "all_pass": True, "results": results,
    }


def _write_campaign_terminal(
    targets: Sequence[ManifestTarget], outcomes: Sequence[TargetOutcome],
    layout: ExecutionLayout,
) -> None:
    summary_path = layout.output_root / "SUMMARY.json"
    summary = _campaign_summary(targets, outcomes, layout)
    _write_json_exclusive(summary_path, summary)
    _write_json_exclusive(layout.output_root / "COMPLETE", {
        "format": CAMPAIGN_COMPLETE_FORMAT,
        "artifact_version": layout.version,
        "campaign_kind": summary["campaign_kind"],
        "ordered_target_ids": summary["ordered_target_ids"],
        "summary_sha256": file_sha256(summary_path),
        "target_interval_sha256": summary["target_interval_sha256"],
        "source_contract_sha256": summary["source_contract_sha256"],
        "target_count": summary["target_count"],
    })


def _run_versioned_campaign(
    targets: Sequence[ManifestTarget], layout: ExecutionLayout,
) -> int:
    if layout.version is None:
        raise ValueError("versioned campaignにはartifact-versionが必要です")
    _validate_versioned_campaign_targets(targets, layout)
    outcomes: list[TargetOutcome] = []
    for target in targets:
        if (layout.output_root / "STOP").exists():
            print("[reserve-background] STOPを検知しました", flush=True)
            break
        outcomes.append(_guard_versioned_target(target, layout))
    if not _campaign_all_passed(targets, outcomes):
        failed = [outcome.target_id for outcome in outcomes if not outcome.contract_pass]
        missing = [target.target_id for target in targets[len(outcomes):]]
        print(f"[reserve-background] campaign failed={failed} missing={missing}", flush=True)
        return 1
    try:
        _write_campaign_terminal(targets, outcomes, layout)
    except Exception as error:  # noqa: BLE001 - 終端証明失敗を成功終了させない
        print(f"[reserve-background] campaign terminal failed: {error}", flush=True)
        return 1
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectではありません: {path}")  # noqa: TRY004
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
    ) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = _parse_args()
    _validate_versioned_campaign_request(args.target, args.artifact_version)
    targets = reserve_targets(args.target)
    layout = execution_layout(args.artifact_version)
    specs = _campaign_specs_for_targets(targets)
    try:
        _prepare_claim(layout, targets)
        print(
            f"[reserve-background] queued={len(targets)} "
            f"excluded={sorted(EXCLUDED_TARGETS)}", flush=True,
        )
        if layout.version is not None:
            return _run_versioned_campaign(targets, layout)
        for target in targets:
            if (layout.output_root / "STOP").exists():
                print("[reserve-background] STOPを検知しました", flush=True)
                break
            _process_target(target, layout)
        return 0
    finally:
        _release_campaign_locks(specs)


if __name__ == "__main__":
    raise SystemExit(main())
