"""保持側の1枠逐次解析が暫定30本と混ざらないことを検証する。"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_event_reserve_background_v1 as reserve
from scripts.run_event_pilot_48_v1 import ManifestTarget, VideoMetadata
from scripts.run_event_reserve_background_v1 import reserve_targets, target_paths

FPS = 60.0
FRAME_COUNT = 1_200_000
VIDEO_METADATA = VideoMetadata(FRAME_COUNT, FPS)


@pytest.fixture(autouse=True)
def _release_campaign_locks_after_test(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """同一pytest process内では各テストを別CLI processとして隔離する。"""

    reserve._release_c96_campaign_lock()
    reserve._release_c36_c60_campaign_lock()
    monkeypatch.setattr(reserve, "_source_video_sha256", lambda _target: "f" * 64)
    yield
    reserve._release_c96_campaign_lock()
    reserve._release_c36_c60_campaign_lock()


def _targets_in_order(target_ids: tuple[str, ...]) -> tuple[ManifestTarget, ...]:
    """台帳順へ正規化せず、テスト指定順のtarget列を作る。"""

    known = {target.target_id: target for target in reserve_targets()}
    return tuple(known[target_id] for target_id in target_ids)


def test_background_queue_contains_only_reserve_and_excludes_review_candidate() -> None:
    targets = reserve_targets()

    assert len(targets) == 48
    assert all(target.subset == "reserve" for target in targets)
    assert "c109" not in {target.target_id for target in targets}


def test_background_queue_can_be_narrowed_without_changing_manifest_order() -> None:
    targets = reserve_targets(("51", "35"))

    assert [target.target_id for target in targets] == ["35", "51"]


def test_background_queue_rejects_pilot_and_c109_targets() -> None:
    with pytest.raises(ValueError, match="保持側"):
        reserve_targets(("38",))
    with pytest.raises(ValueError, match="保持側"):
        reserve_targets(("c109",))


def test_background_paths_are_separate_from_first30_outputs() -> None:
    paths = target_paths(reserve_targets(("35",))[0])

    assert "event_reserve_background_2026-08-31" in paths.work_npz.as_posix()
    assert "event_first30" not in paths.work_npz.as_posix()
    assert paths.work_npz.name == "video_35.npz"
    assert paths.result_json.name == "video_35_result.json"
    assert paths.attempt_id == "event-reserve-background-v1-35-20260831"


def test_clip_targets_use_target_id_unique_artifact_names() -> None:
    targets = reserve_targets(("c96s1", "c96s2", "c96s3"))
    paths = tuple(target_paths(target) for target in targets)

    assert len({item.video for item in paths}) == 1
    for field in (
        "calibration", "calibration_log", "work_npz", "collection_log",
        "launcher_log", "result_json",
    ):
        values = tuple(getattr(item, field) for item in paths)
        assert len(set(values)) == 3
        assert all(item.target_id in value.name for item, value in zip(paths, values))


def _patch_layout_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reserve, "OUTPUT_ROOT", tmp_path / "old-output")
    monkeypatch.setattr(reserve, "CALIBRATION_ROOT", tmp_path / "old-calibration")
    monkeypatch.setattr(reserve, "LOG_ROOT", tmp_path / "old-logs")
    monkeypatch.setattr(reserve, "RAW_VIDEO_ROOT", tmp_path / "raw-videos")


def test_explicit_artifact_version_separates_every_write_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c96-clipfix-v1")
    paths = target_paths(reserve_targets(("c96s1",))[0], layout)

    assert layout.output_root.name == "old-output__version=c96-clipfix-v1"
    assert layout.calibration_root.name == "old-calibration__version=c96-clipfix-v1"
    assert layout.log_root.name == "old-logs__version=c96-clipfix-v1"
    assert layout.claim_root.parent == layout.output_root
    assert paths.output_root == layout.output_root
    assert paths.attempt_id.startswith("event-reserve-background-v1-c96-clipfix-v1-")
    assert paths.calibration.is_relative_to(layout.calibration_root)
    assert paths.collection_log.is_relative_to(layout.log_root)


def test_explicit_artifact_version_requires_targets() -> None:
    with pytest.raises(SystemExit):
        reserve._parse_args(("--artifact-version", "c96-clipfix-v1"))


def test_explicit_artifact_version_rejects_path_syntax() -> None:
    with pytest.raises(ValueError, match="安全なASCII"):
        reserve.execution_layout("../escape")


def test_versioned_queue_claim_is_written_below_new_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c96-clipfix-v1")
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)

    reserve._prepare_claim(layout, targets)

    claim = json.loads((layout.claim_root / "CLAIM.json").read_text(encoding="utf-8"))
    assert claim["artifact_version"] == "c96-clipfix-v1"
    assert not reserve.OUTPUT_ROOT.exists()
    assert not reserve.CALIBRATION_ROOT.exists()
    assert not reserve.LOG_ROOT.exists()


@pytest.mark.parametrize(
    "target_ids",
    (
        ("c96s1",),
        ("c96s1", "c96s2"),
        ("c96s1", "c96s2", "c96s2", "c96s3"),
        ("c96s2", "c96s1", "c96s3"),
        ("c96s1", "c96s2", "c96s3", "35"),
        ("c36",),
        ("c60", "c36"),
        ("c36", "c36", "c60"),
        ("c36", "c60", "35"),
        ("35",),
    ),
)
def test_versioned_c96_main_rejects_non_exact_request_before_setup(
    monkeypatch: pytest.MonkeyPatch, target_ids: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        reserve, "_parse_args",
        lambda: SimpleNamespace(target=list(target_ids), artifact_version="c96-fix-v1"),
    )
    monkeypatch.setattr(
        reserve, "reserve_targets",
        lambda _requested: pytest.fail("不正request後に台帳を解決してはいけません"),
    )

    with pytest.raises(ValueError, match="この順で全件"):
        reserve.main()


@pytest.mark.parametrize(
    "target_ids",
    (
        ("c96s1",),
        ("c96s1", "c96s2"),
        ("c96s1", "c96s2", "c96s2", "c96s3"),
        ("c96s2", "c96s1", "c96s3"),
        ("c96s1", "c96s2", "c96s3", "35"),
        ("35",),
    ),
)
def test_versioned_c96_claim_rejects_non_exact_targets_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_ids: tuple[str, ...],
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c96-fix-v1")
    targets = _targets_in_order(target_ids)

    with pytest.raises(ValueError, match="exact ordered complete set"):
        reserve._prepare_claim(layout, targets)

    assert not layout.output_root.exists()
    assert not layout.calibration_root.exists()
    assert not layout.log_root.exists()
    assert not reserve._fixed_campaign_claim_path().exists()


def test_both_fixed_versioned_campaign_requests_are_accepted() -> None:
    assert reserve._validate_versioned_campaign_request(
        reserve.C96_RECOVERY_TARGET_IDS, "c96-v1",
    ) is reserve.C96_CAMPAIGN
    assert reserve._validate_versioned_campaign_request(
        reserve.C36_C60_RECOVERY_TARGET_IDS, "c36-c60-v1",
    ) is reserve.C36_C60_CAMPAIGN


@pytest.mark.parametrize(
    "target_ids",
    (
        ("c36",),
        ("c60",),
        ("c36", "c36", "c60"),
        ("c60", "c36"),
        ("c36", "c60", "35"),
        ("c36", "c96s1", "c60"),
    ),
)
def test_versioned_c36_c60_rejects_non_exact_set_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_ids: tuple[str, ...],
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c36-c60-invalid-v1")
    targets = _targets_in_order(target_ids)

    with pytest.raises(ValueError, match="exact ordered complete set"):
        reserve._prepare_claim(layout, targets)

    assert not layout.output_root.exists()
    assert not layout.calibration_root.exists()
    assert not layout.log_root.exists()
    assert not reserve._campaign_claim_path(reserve.C36_C60_CAMPAIGN).exists()


def test_versioned_c36_c60_rejects_clip_manifest_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c36-c60-clipped-v1")
    original = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    targets = (replace(original[0], clip_start_sec="1", clip_end_sec="2"), original[1])

    with pytest.raises(ValueError, match="clip指定なし・全長処理"):
        reserve._prepare_claim(layout, targets)

    assert not layout.output_root.exists()
    assert not reserve._campaign_claim_path(reserve.C36_C60_CAMPAIGN).exists()


@pytest.mark.parametrize("competing_version", [None, "c96-clipfix-v2"])
def test_fixed_c96_claim_blocks_other_version_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    competing_version: str | None,
) -> None:
    """実行中lockはartifact namespaceを変えても同時再解析を許さない。"""

    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(("c96s1", "c96s2", "c96s3"))
    first = reserve.execution_layout("c96-clipfix-v1")
    reserve._prepare_claim(first, targets)

    fixed = reserve._fixed_campaign_claim_path()
    attempts = tuple((fixed / reserve.C96_CAMPAIGN_ATTEMPTS_DIR).glob("*.json"))
    assert len(attempts) == 1
    claim = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert claim["ordered_target_ids"] == ["c96s1", "c96s2", "c96s3"]
    assert claim["target_interval_sha256"] == reserve._target_interval_sha256(targets)
    assert not fixed.is_relative_to(first.output_root)

    competing = reserve.execution_layout(competing_version)
    with pytest.raises(FileExistsError, match="処理中"):
        reserve._prepare_claim(competing, targets)
    assert not competing.output_root.exists()


def test_fixed_claim_history_allows_retry_after_process_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """過去claimを保持したまま別versionで再試行できる。"""

    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    fixed = reserve._fixed_campaign_claim_path()
    fixed.mkdir(parents=True)
    legacy = fixed / "CLAIM.json"
    legacy.write_text('{"legacy": true}\n', encoding="utf-8")
    legacy_bytes = legacy.read_bytes()

    first = reserve.execution_layout("c96-retry-v1")
    reserve._prepare_claim(first, targets)
    reserve._release_c96_campaign_lock()
    second = reserve.execution_layout("c96-retry-v2")
    reserve._prepare_claim(second, targets)

    attempts = tuple((fixed / reserve.C96_CAMPAIGN_ATTEMPTS_DIR).glob("*.json"))
    versions = {
        json.loads(path.read_text(encoding="utf-8"))["artifact_version"]
        for path in attempts
    }
    assert len(attempts) == 2
    assert legacy.read_bytes() == legacy_bytes
    assert versions == {"c96-retry-v1", "c96-retry-v2"}


def test_c96_lock_blocks_concurrent_process_and_releases_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """別processの異常終了後も永久claim化せず再取得できる。"""

    _patch_layout_roots(tmp_path, monkeypatch)
    child_code = """from pathlib import Path
import sys, time
import scripts.run_event_reserve_background_v1 as reserve
reserve.OUTPUT_ROOT = Path(sys.argv[1]) / 'old-output'
reserve._acquire_c96_campaign_lock()
print('locked', flush=True)
time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "locked"
    try:
        with pytest.raises(FileExistsError, match="処理中"):
            reserve._acquire_c96_campaign_lock()
    finally:
        child.kill()
        child.wait(timeout=10)

    reserve._acquire_c96_campaign_lock()


@pytest.mark.parametrize("competing_version", [None, "c36-c60-full-v2"])
def test_c36_c60_lock_blocks_other_version_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    competing_version: str | None,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    campaign_targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    first = reserve.execution_layout("c36-c60-full-v1")
    reserve._prepare_claim(first, campaign_targets)

    competing = reserve.execution_layout(competing_version)
    default_targets = reserve_targets(("c36",)) if competing_version is None else campaign_targets
    with pytest.raises(FileExistsError, match="処理中"):
        reserve._prepare_claim(competing, default_targets)
    assert not competing.output_root.exists()


def test_c36_c60_attempt_claim_is_append_only_across_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    fixed = reserve._campaign_claim_path(reserve.C36_C60_CAMPAIGN)
    fixed.mkdir(parents=True)
    legacy = fixed / "CLAIM.json"
    legacy.write_bytes(b'{"legacy": true}\n')
    before = legacy.read_bytes()

    reserve._prepare_claim(reserve.execution_layout("c36-c60-retry-v1"), targets)
    reserve._release_c36_c60_campaign_lock()
    reserve._prepare_claim(reserve.execution_layout("c36-c60-retry-v2"), targets)

    attempts = tuple((fixed / reserve.C96_CAMPAIGN_ATTEMPTS_DIR).glob("*.json"))
    values = [json.loads(path.read_text(encoding="utf-8")) for path in attempts]
    assert legacy.read_bytes() == before
    assert {value["artifact_version"] for value in values} == {
        "c36-c60-retry-v1", "c36-c60-retry-v2",
    }
    assert all(value["campaign_kind"] == reserve.C36_C60_CAMPAIGN_KIND for value in values)


def test_c36_c60_lock_releases_after_process_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    child_code = """from pathlib import Path
import sys, time
import scripts.run_event_reserve_background_v1 as reserve
reserve.OUTPUT_ROOT = Path(sys.argv[1]) / 'old-output'
reserve._acquire_c36_c60_campaign_lock()
print('locked', flush=True)
time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "locked"
    try:
        with pytest.raises(FileExistsError, match="処理中"):
            reserve._acquire_c36_c60_campaign_lock()
    finally:
        child.kill()
        child.wait(timeout=10)

    reserve._acquire_c36_c60_campaign_lock()


def test_claim_records_exact_ordered_targets_and_order_sensitive_interval_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = reserve.execution_layout("c96-contract-v1")

    reserve._prepare_claim(layout, targets)

    claim = json.loads((layout.claim_root / "CLAIM.json").read_text(encoding="utf-8"))
    assert claim["ordered_target_ids"] == ["c96s1", "c96s2", "c96s3"]
    assert claim["target_interval_sha256"] == reserve._target_interval_sha256(targets)
    assert reserve._target_interval_sha256(targets) != reserve._target_interval_sha256(
        tuple(reversed(targets))
    )


def test_c36_c60_claim_fixes_full_length_order_kind_and_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    layout = reserve.execution_layout("c36-c60-contract-v1")

    reserve._prepare_claim(layout, targets)

    claim = json.loads((layout.claim_root / "CLAIM.json").read_text(encoding="utf-8"))
    assert claim["campaign_kind"] == reserve.C36_C60_CAMPAIGN_KIND
    assert claim["ordered_target_ids"] == ["c36", "c60"]
    assert claim["target_interval_sha256"] == reserve._target_interval_sha256(targets)
    assert claim["target_intervals"] == [
        {
            "target_id": target.target_id,
            "source_video_path": target.source_video_path.as_posix(),
            "clip_start_sec": None,
            "clip_end_sec": None,
        }
        for target in targets
    ]
    assert len(claim["source_contract_sha256"]) == 64
    assert [source["source_video_sha256"] for source in claim["sources"]] == [
        "f" * 64, "f" * 64,
    ]


def test_explicit_artifact_version_rejects_existing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout("c96-clipfix-v1")
    layout.output_root.mkdir()

    with pytest.raises(FileExistsError, match="既に存在"):
        reserve.execution_layout("c96-clipfix-v1")


def test_versioned_layout_rejects_raw_root_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    layout = reserve.ExecutionLayout(
        raw / "child", tmp_path / "calibration", tmp_path / "logs",
        raw / "child" / "claim", "attempt-v2", "v2",
    )
    monkeypatch.setattr(reserve, "RAW_VIDEO_ROOT", raw)
    with pytest.raises(ValueError, match="raw動画root"):
        reserve._validate_new_layout(layout)


def _sandbox_collection_paths(paths: reserve.TargetPaths, tmp_path: Path) -> reserve.TargetPaths:
    return replace(
        paths,
        work_npz=tmp_path / "work" / paths.work_npz.name,
        collection_log=tmp_path / paths.collection_log.name,
        launcher_log=tmp_path / paths.launcher_log.name,
        result_json=tmp_path / paths.result_json.name,
        runs=tmp_path / "runs",
    )


@pytest.mark.parametrize(
    ("target_id", "expected_start", "expected_max"),
    (
        ("c96s1", "915.0", "3201.0"),
        ("c96s2", "7242.0", "3405.0"),
        ("c96s3", "12570.0", "3702.0"),
    ),
)
def test_clip_collection_command_uses_manifest_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_id: str, expected_start: str, expected_max: str,
) -> None:
    paths = _sandbox_collection_paths(
        target_paths(reserve_targets((target_id,))[0]), tmp_path,
    )
    captured: list[tuple[str, ...]] = []
    monkeypatch.setattr(reserve, "_emit_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reserve, "_run_logged",
        lambda command, _log: captured.append(tuple(command)) or 0,
    )

    assert reserve._run_target(paths, VIDEO_METADATA) == 0

    command = captured[0]
    assert command[command.index("--start-sec") + 1] == expected_start
    assert command[command.index("--max-sec") + 1] == expected_max


def test_full_length_collection_command_keeps_zero_and_full_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _sandbox_collection_paths(
        target_paths(reserve_targets(("35",))[0]), tmp_path,
    )
    captured: list[tuple[str, ...]] = []
    monkeypatch.setattr(reserve, "_emit_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reserve, "_run_logged",
        lambda command, _log: captured.append(tuple(command)) or 0,
    )

    assert reserve._run_target(paths, VIDEO_METADATA) == 0

    command = captured[0]
    expected_max = reserve.full_length_max_sec(VIDEO_METADATA)
    assert command[command.index("--start-sec") + 1] == "0"
    assert command[command.index("--max-sec") + 1] == repr(expected_max)


@pytest.mark.parametrize("target_id", reserve.C36_C60_RECOVERY_TARGET_IDS)
def test_c36_c60_collection_command_is_explicitly_full_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_id: str,
) -> None:
    paths = _sandbox_collection_paths(
        target_paths(reserve_targets((target_id,))[0]), tmp_path,
    )
    captured: list[tuple[str, ...]] = []
    monkeypatch.setattr(reserve, "_emit_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reserve, "_run_logged",
        lambda command, _log: captured.append(tuple(command)) or 0,
    )

    assert paths.clip_start_sec is None
    assert paths.clip_end_sec is None
    assert reserve._run_target(paths, VIDEO_METADATA) == 0
    command = captured[0]
    assert command[command.index("--start-sec") + 1] == "0"
    assert command[command.index("--max-sec") + 1] == repr(
        reserve.full_length_max_sec(VIDEO_METADATA)
    )


def test_clip_calibration_window_stays_inside_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = target_paths(reserve_targets(("c96s1",))[0])
    paths = replace(
        original,
        calibration=tmp_path / "c96s1.json",
        calibration_log=tmp_path / "c96s1_calibration.log",
    )
    captured: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], _log: Path) -> int:
        captured.append(command)
        paths.calibration.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(reserve, "_emit_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(reserve, "_run_logged", fake_run)

    assert reserve._ensure_calibration(paths, VIDEO_METADATA)
    command = captured[0]
    start = float(command[command.index("--start-sec") + 1])
    window = float(command[command.index("--window-sec") + 1])
    assert start >= paths.clip_start_sec
    assert start + window <= paths.clip_end_sec
    assert (start, window) == (1095.0, 600.0)


def _audit_paths(tmp_path: Path) -> reserve.TargetPaths:
    return reserve.TargetPaths(
        target_id="c96s1",
        video=tmp_path / "data" / "frames" / "video_c96.mp4",
        calibration=tmp_path / "c96s1.json",
        calibration_log=tmp_path / "c96s1_calibration.log",
        work_npz=tmp_path / "work" / "c96s1.npz",
        collection_log=tmp_path / "c96s1_collection.log",
        launcher_log=tmp_path / "c96s1_launcher.log",
        result_json=tmp_path / "c96s1_result.json",
        runs=tmp_path / "runs",
        audit_json=tmp_path / "AUDIT.json",
        attempt_id="event-reserve-background-v1-c96s1-20260831",
        clip_start_sec=915.0,
        clip_end_sec=4116.0,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(
    paths: reserve.TargetPaths,
    *, attempt_id: str | None = None,
    start_frame: int = 54_900,
    end_frame: int = 246_960,
    suffix: str = "one",
    source_path: str = "data/frames/video_c96.mp4",
    source_sha256: str = "a" * 64,
) -> Path:
    run_dir = paths.runs / "schema=v1" / f"video={paths.video.stem}" / f"build={suffix}" / (
        f"attempt={attempt_id or paths.attempt_id}"
    )
    _write_json(run_dir / "manifest.json", {
        "attempt_id": attempt_id or paths.attempt_id,
        "source": {
            "source_video_id": paths.video.stem,
            "source_video_path": source_path,
            "source_video_sha256": source_sha256,
            "frame_count": FRAME_COUNT,
            "processing_start_frame": start_frame,
            "processing_end_frame_exclusive": end_frame,
        },
    })
    return run_dir


def _write_result(paths: reserve.TargetPaths, run_dir: Path, end_frame: int = 246_960) -> None:
    _write_json(paths.result_json, {
        "run_dir": str(run_dir),
        "requested_end_frame_exclusive": end_frame,
        "processed_end_frame_exclusive": end_frame,
    })


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch, paths: reserve.TargetPaths,
) -> None:
    monkeypatch.setattr(reserve, "PROJECT_ROOT", paths.video.parents[2])
    monkeypatch.setattr(reserve, "probe_video", lambda _path: VIDEO_METADATA)


def _patch_successful_audit(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path,
    calls: dict[str, int] | None = None,
) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    def result(name: str, value: object) -> Callable[..., object]:
        def wrapped(*_args: object) -> object:
            if calls is not None:
                calls[name] = calls.get(name, 0) + 1
            return value
        return wrapped

    monkeypatch.setattr(reserve, "file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        reserve, "validate_completed_run",
        result("completed", SimpleNamespace(valid=True, error=None)),
    )
    monkeypatch.setattr(
        reserve, "load_prevalidated_run_events", result("load", (manifest, [])),
    )
    monkeypatch.setattr(
        reserve, "validate_accounting_pilot_loaded_run",
        result("accounting", {
            "conservation_residual_nonzero_count": 0,
            "conservation_residual_observation_side_count": 0,
        }),
    )
    monkeypatch.setattr(
        reserve, "validate_observation_pilot_loaded_run",
        result("observation", {"validation_pass": True}),
    )
    monkeypatch.setattr(
        reserve, "cross_validate_exchange_events_v1",
        result("cross", SimpleNamespace(physical_gate_pass=True)),
    )
    monkeypatch.setattr(
        reserve, "audit_exchange_cross_quarantine_v1",
        result("quarantine", {"unsupported_game_count": 0, "contract_pass": True}),
    )


def test_audit_rejects_run_from_another_target_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    foreign = tmp_path / "other-target" / "attempt=foreign"
    _write_result(paths, foreign)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="runs配下"):
        reserve._audit_completed(paths)


def test_audit_rejects_wrong_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths, attempt_id="wrong-attempt")
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="attempt_id"):
        reserve._audit_completed(paths)


def test_audit_rejects_wrong_manifest_processing_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths, start_frame=0, end_frame=FRAME_COUNT)
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="処理範囲"):
        reserve._audit_completed(paths)


def test_audit_rejects_wrong_result_processing_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths)
    _write_result(paths, run_dir, end_frame=FRAME_COUNT)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="requested_end_frame_exclusive"):
        reserve._audit_completed(paths)


def test_audit_rejects_duplicate_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths, suffix="one")
    _write_run(paths, suffix="two")
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="一意"):
        reserve._audit_completed(paths)


def test_existing_audit_is_revalidated_instead_of_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths, start_frame=0, end_frame=FRAME_COUNT)
    _write_json(paths.audit_json, {
        "target_id": paths.target_id,
        "run_dir": str(run_dir),
        "contract_pass": True,
    })
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="処理範囲"):
        reserve._audit_completed(paths)


def test_exact_clip_run_is_audited_and_records_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths)
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)
    _patch_successful_audit(monkeypatch, run_dir)

    assert reserve._audit_completed(paths)
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    assert audit["attempt_id"] == paths.attempt_id
    assert audit["processing_start_frame"] == 54_900
    assert audit["processing_end_frame_exclusive"] == 246_960


def test_c36_full_length_audit_records_absent_clip_and_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = replace(
        _audit_paths(tmp_path),
        target_id="c36",
        video=tmp_path / "data" / "frames" / "video_c36.mp4",
        audit_json=tmp_path / "c36-AUDIT.json",
        attempt_id="event-reserve-background-v1-c36-20260831",
        clip_start_sec=None,
        clip_end_sec=None,
    )
    run_dir = _write_run(
        paths, start_frame=0, end_frame=FRAME_COUNT,
        source_path="data/frames/video_c36.mp4",
    )
    _write_result(paths, run_dir, end_frame=FRAME_COUNT)
    _patch_probe(monkeypatch, paths)
    _patch_successful_audit(monkeypatch, run_dir)

    assert reserve._audit_completed(paths)
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    assert audit["clip_start_sec"] is None
    assert audit["clip_end_sec"] is None
    assert audit["source_video_path"] == "data/frames/video_c36.mp4"
    assert audit["source_video_sha256"] == "a" * 64
    assert audit["processing_start_frame"] == 0
    assert audit["processing_end_frame_exclusive"] == FRAME_COUNT


def test_existing_audit_reruns_all_independent_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths)
    _write_json(paths.audit_json, {
        "target_id": paths.target_id,
        "run_dir": str(run_dir),
        "contract_pass": True,
    })
    calls: dict[str, int] = {}
    _patch_probe(monkeypatch, paths)
    _patch_successful_audit(monkeypatch, run_dir, calls)

    assert reserve._audit_completed(paths)
    assert calls == {
        "completed": 1, "load": 1, "accounting": 1,
        "observation": 1, "cross": 1, "quarantine": 1,
    }


def test_saved_pass_cannot_override_fresh_audit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths)
    _write_json(paths.audit_json, {
        "target_id": paths.target_id,
        "run_dir": str(run_dir),
        "contract_pass": True,
    })
    _patch_probe(monkeypatch, paths)
    _patch_successful_audit(monkeypatch, run_dir)
    monkeypatch.setattr(
        reserve, "audit_exchange_cross_quarantine_v1",
        lambda _report: {"unsupported_game_count": 1, "contract_pass": False},
    )

    with pytest.raises(ValueError, match="独立再監査"):
        reserve._audit_completed(paths)


def test_audit_rejects_source_path_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths, source_path="data/frames/other.mp4")
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)

    with pytest.raises(ValueError, match="動画または処理範囲"):
        reserve._audit_completed(paths)


def test_audit_rejects_source_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _audit_paths(tmp_path)
    run_dir = _write_run(paths)
    _write_result(paths, run_dir)
    _patch_probe(monkeypatch, paths)
    monkeypatch.setattr(reserve, "file_sha256", lambda _path: "b" * 64)

    with pytest.raises(ValueError, match="SHA-256"):
        reserve._audit_completed(paths)


def _versioned_campaign_layout(tmp_path: Path) -> reserve.ExecutionLayout:
    output = tmp_path / "campaign-output"
    return reserve.ExecutionLayout(
        output, tmp_path / "campaign-calibration", tmp_path / "campaign-logs",
        output / "active-queue-claim", "attempt-campaign", "campaign-v1",
    )


def _audited_outcome(
    target: ManifestTarget, layout: reserve.ExecutionLayout,
) -> reserve.TargetOutcome:
    paths = target_paths(target, layout)
    interval = reserve._target_interval_value(target)
    run_dir = paths.runs / "schema=v1" / f"video={paths.video.stem}" / (
        f"attempt={paths.attempt_id}"
    )
    _write_json(run_dir / "manifest.json", {"artifact": "manifest"})
    _write_json(run_dir / "validation.json", {"artifact": "validation"})
    _write_json(run_dir / "COMPLETE", {"artifact": "complete"})
    _write_json(run_dir / "events" / "part-00000.jsonl", {"artifact": "events"})
    _write_json(paths.audit_json, {
        "target_id": target.target_id,
        "contract_pass": True,
        "run_dir": str(run_dir.resolve()),
        "source_video_path": interval["source_video_path"],
        "source_video_sha256": "f" * 64,
        "clip_start_sec": interval["clip_start_sec"],
        "clip_end_sec": interval["clip_end_sec"],
    })
    return reserve._freeze_versioned_outcome(paths, "audit_passed")


def _patch_terminal_audit(
    monkeypatch: pytest.MonkeyPatch, calls: list[str] | None = None,
) -> None:
    monkeypatch.setattr(reserve, "probe_video", lambda _path: VIDEO_METADATA)

    def fresh(
        paths: reserve.TargetPaths, _run_dir: Path, _metadata: VideoMetadata,
    ) -> dict[str, object]:
        if calls is not None:
            calls.append(paths.target_id)
        return json.loads(paths.audit_json.read_text(encoding="utf-8"))

    monkeypatch.setattr(reserve, "_independent_audit", fresh)


def _write_versioned_claim(
    layout: reserve.ExecutionLayout, targets: tuple[ManifestTarget, ...],
) -> None:
    spec = reserve._validate_versioned_campaign_targets(targets, layout)
    assert spec is not None
    value = {
        "format": reserve.QUEUE_FORMAT,
        "artifact_version": layout.version,
        **reserve._campaign_contract(spec, targets),
    }
    _write_json(layout.claim_root / "CLAIM.json", value)


@pytest.mark.parametrize(
    "target_ids",
    (
        ("c96s1", "c96s2"),
        ("c96s1", "c96s1", "c96s2", "c96s3"),
        ("c96s3", "c96s2", "c96s1"),
        ("c96s1", "c96s2", "c96s3", "35"),
        ("35",),
    ),
)
def test_versioned_campaign_rejects_non_exact_set_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_ids: tuple[str, ...],
) -> None:
    targets = _targets_in_order(target_ids)
    layout = _versioned_campaign_layout(tmp_path)
    monkeypatch.setattr(
        reserve, "_guard_versioned_target",
        lambda *_args: pytest.fail("不完全campaignのtargetを処理してはいけません"),
    )

    with pytest.raises(ValueError, match="exact ordered complete set"):
        reserve._run_versioned_campaign(targets, layout)

    assert not layout.output_root.exists()
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def test_versioned_campaign_writes_summary_and_complete_only_after_all_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)
    _write_versioned_claim(layout, targets)
    _patch_terminal_audit(monkeypatch)
    monkeypatch.setattr(reserve, "_guard_versioned_target", _audited_outcome)

    assert reserve._run_versioned_campaign(targets, layout) == 0

    summary_path = layout.output_root / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    complete = json.loads((layout.output_root / "COMPLETE").read_text(encoding="utf-8"))
    assert summary["ordered_target_ids"] == ["c96s1", "c96s2", "c96s3"]
    assert summary["all_pass"] is True
    assert [row["contract_pass"] for row in summary["results"]] == [True, True, True]
    assert complete["summary_sha256"] == reserve.file_sha256(summary_path)
    assert complete["target_interval_sha256"] == summary["target_interval_sha256"]


def test_c36_c60_campaign_is_serial_isolated_and_terminal_contract_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    for root in (reserve.OUTPUT_ROOT, reserve.CALIBRATION_ROOT, reserve.LOG_ROOT):
        root.mkdir(parents=True)
        (root / "OLD.txt").write_bytes(b"do-not-touch")
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    layout = reserve.execution_layout("c36-c60-full-v1")
    reserve._prepare_claim(layout, targets)
    order: list[str] = []
    terminal_calls: list[str] = []
    _patch_terminal_audit(monkeypatch, terminal_calls)

    def audited(
        target: ManifestTarget, selected_layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        order.append(target.target_id)
        return _audited_outcome(target, selected_layout)

    monkeypatch.setattr(reserve, "_guard_versioned_target", audited)
    assert reserve._run_versioned_campaign(targets, layout) == 0

    assert order == ["c36", "c60"]
    assert terminal_calls == ["c36", "c60"]
    summary_path = layout.output_root / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    complete = json.loads((layout.output_root / "COMPLETE").read_text(encoding="utf-8"))
    assert summary["campaign_kind"] == reserve.C36_C60_CAMPAIGN_KIND
    assert summary["ordered_target_ids"] == ["c36", "c60"]
    assert summary["target_count"] == 2
    assert complete["campaign_kind"] == summary["campaign_kind"]
    assert complete["ordered_target_ids"] == summary["ordered_target_ids"]
    assert complete["target_interval_sha256"] == summary["target_interval_sha256"]
    assert complete["source_contract_sha256"] == summary["source_contract_sha256"]
    for target in targets:
        paths = target_paths(target, layout)
        assert paths.work_npz.is_relative_to(layout.output_root)
        assert paths.runs.is_relative_to(layout.output_root)
        assert paths.audit_json.is_relative_to(layout.output_root)
        assert paths.calibration.is_relative_to(layout.calibration_root)
        assert paths.collection_log.is_relative_to(layout.log_root)
    for root in (reserve.OUTPUT_ROOT, reserve.CALIBRATION_ROOT, reserve.LOG_ROOT):
        assert (root / "OLD.txt").read_bytes() == b"do-not-touch"


def test_c36_c60_campaign_failure_has_no_terminal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)
    _write_versioned_claim(layout, targets)
    _patch_terminal_audit(monkeypatch)

    def outcome(
        target: ManifestTarget, _layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        passed = target.target_id == "c36"
        return reserve.TargetOutcome(target.target_id, "done" if passed else "failed", passed)

    monkeypatch.setattr(reserve, "_guard_versioned_target", outcome)
    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def test_c36_c60_campaign_rejects_audit_with_clip_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)
    _write_versioned_claim(layout, targets)
    _patch_terminal_audit(monkeypatch)

    def clipped_audit(
        target: ManifestTarget, selected_layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        outcome = _audited_outcome(target, selected_layout)
        if target.target_id == "c60":
            audit_path = target_paths(target, selected_layout).audit_json
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["clip_start_sec"] = 1.0
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return outcome

    monkeypatch.setattr(reserve, "_guard_versioned_target", clipped_audit)
    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def _mutate_frozen_campaign_artifact(paths: reserve.TargetPaths, mutation: str) -> None:
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    run_dir = Path(audit["run_dir"])
    if mutation == "audit_rewrite":
        audit["untrusted_extra"] = "same-six-fields"
        paths.audit_json.write_text(json.dumps(audit), encoding="utf-8")
    elif mutation == "audit_run_path":
        audit["run_dir"] = str((paths.runs / "foreign-run").resolve())
        paths.audit_json.write_text(json.dumps(audit), encoding="utf-8")
    elif mutation == "complete_delete":
        (run_dir / "COMPLETE").unlink()
    elif mutation == "manifest_tamper":
        (run_dir / "manifest.json").write_bytes(b'{"tampered":true}')
    elif mutation == "validation_delete":
        (run_dir / "validation.json").unlink()
    elif mutation == "events_tamper":
        (run_dir / "events" / "part-00000.jsonl").write_bytes(b'{"tampered":true}')
    else:
        raise AssertionError(f"未対応mutationです: {mutation}")


@pytest.mark.parametrize(
    "target_ids",
    (reserve.C36_C60_RECOVERY_TARGET_IDS, reserve.C96_RECOVERY_TARGET_IDS),
)
@pytest.mark.parametrize(
    "mutation",
    (
        "audit_rewrite", "audit_run_path", "complete_delete",
        "manifest_tamper", "validation_delete", "events_tamper",
    ),
)
def test_terminal_reaudit_rejects_post_processing_original_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target_ids: tuple[str, ...], mutation: str,
) -> None:
    targets = reserve_targets(target_ids)
    layout = _versioned_campaign_layout(tmp_path)
    _write_versioned_claim(layout, targets)
    outcomes = {
        target.target_id: _audited_outcome(target, layout) for target in targets
    }
    first_paths = target_paths(targets[0], layout)
    original_audit = json.loads(first_paths.audit_json.read_text(encoding="utf-8"))
    _mutate_frozen_campaign_artifact(first_paths, mutation)
    calls: list[str] = []
    _patch_terminal_audit(monkeypatch, calls)
    monkeypatch.setattr(
        reserve, "_guard_versioned_target",
        lambda target, _layout: outcomes[target.target_id],
    )

    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert calls == list(target_ids)
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()
    if mutation.startswith("audit_"):
        changed = json.loads(first_paths.audit_json.read_text(encoding="utf-8"))
        for key in (
            "source_video_path", "source_video_sha256",
            "clip_start_sec", "clip_end_sec", "contract_pass",
        ):
            assert changed[key] == original_audit[key]


def test_versioned_campaign_failure_is_nonzero_without_terminal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)

    def outcome(
        target: ManifestTarget, _layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        passed = target.target_id != "c96s2"
        return reserve.TargetOutcome(target.target_id, "done" if passed else "failed", passed)

    monkeypatch.setattr(reserve, "_guard_versioned_target", outcome)
    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def test_versioned_campaign_rejects_audit_that_does_not_prove_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)

    def invalid_audit(
        target: ManifestTarget, selected_layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        paths = target_paths(target, selected_layout)
        _write_json(paths.audit_json, {"target_id": target.target_id, "contract_pass": False})
        return reserve.TargetOutcome(target.target_id, "audit_passed", True)

    monkeypatch.setattr(reserve, "_guard_versioned_target", invalid_audit)
    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def test_versioned_campaign_stop_is_missing_and_has_no_terminal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)
    layout.output_root.mkdir(parents=True)
    (layout.output_root / "STOP").write_text("stop\n", encoding="utf-8")
    monkeypatch.setattr(
        reserve, "_guard_versioned_target",
        lambda *_args: pytest.fail("STOP後にtargetを開始してはいけません"),
    )

    assert reserve._run_versioned_campaign(targets, layout) == 1
    assert not (layout.output_root / "SUMMARY.json").exists()
    assert not (layout.output_root / "COMPLETE").exists()


def test_versioned_main_propagates_campaign_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(reserve.C96_RECOVERY_TARGET_IDS)
    layout = _versioned_campaign_layout(tmp_path)
    claimed: list[tuple[str, ...]] = []
    prepare_claim = reserve._prepare_claim
    monkeypatch.setattr(
        reserve, "_parse_args",
        lambda: SimpleNamespace(
            target=list(reserve.C96_RECOVERY_TARGET_IDS), artifact_version="campaign-v1",
        ),
    )
    monkeypatch.setattr(reserve, "reserve_targets", lambda _requested: targets)
    monkeypatch.setattr(reserve, "execution_layout", lambda _version: layout)

    def prepare(
        selected_layout: reserve.ExecutionLayout,
        selected: tuple[ManifestTarget, ...],
    ) -> None:
        claimed.append(tuple(item.target_id for item in selected))
        prepare_claim(selected_layout, selected)

    monkeypatch.setattr(
        reserve, "_prepare_claim", prepare,
    )
    monkeypatch.setattr(reserve, "_run_versioned_campaign", lambda *_args: 1)

    assert reserve.main() == 1
    assert claimed == [("c96s1", "c96s2", "c96s3")]
    reserve._acquire_c96_campaign_lock()


def test_default_main_keeps_legacy_zero_exit_on_target_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(("35",))
    layout = replace(_versioned_campaign_layout(tmp_path), version=None)
    monkeypatch.setattr(
        reserve, "_parse_args", lambda: SimpleNamespace(target=["35"], artifact_version=None),
    )
    monkeypatch.setattr(reserve, "reserve_targets", lambda _requested: targets)
    monkeypatch.setattr(reserve, "execution_layout", lambda _version: layout)
    monkeypatch.setattr(reserve, "_prepare_claim", lambda *_args: None)
    monkeypatch.setattr(
        reserve, "_process_target",
        lambda target, _layout: reserve.TargetOutcome(target.target_id, "failed", False),
    )

    assert reserve.main() == 0


def test_default_missing_c36_c60_skips_new_hash_claim_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(reserve.C36_C60_RECOVERY_TARGET_IDS)
    _patch_layout_roots(tmp_path, monkeypatch)
    layout = reserve.execution_layout()
    monkeypatch.setattr(reserve, "PROJECT_ROOT", tmp_path / "missing-input-root")
    monkeypatch.setattr(
        reserve, "_source_video_sha256",
        lambda _target: pytest.fail("defaultでc36/c60 source hashを読んではいけません"),
    )
    monkeypatch.setattr(
        reserve, "_parse_args",
        lambda: SimpleNamespace(
            target=list(reserve.C36_C60_RECOVERY_TARGET_IDS), artifact_version=None,
        ),
    )
    monkeypatch.setattr(reserve, "reserve_targets", lambda _requested: targets)
    monkeypatch.setattr(reserve, "execution_layout", lambda _version: layout)
    original_process = reserve._process_target
    outcomes: list[reserve.TargetOutcome] = []

    def process(
        target: ManifestTarget, selected_layout: reserve.ExecutionLayout,
    ) -> reserve.TargetOutcome:
        outcome = original_process(target, selected_layout)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(reserve, "_process_target", process)
    assert reserve.main() == 0
    assert [outcome.target_id for outcome in outcomes] == ["c36", "c60"]
    assert all(outcome.state == "input_or_calibration_failed" for outcome in outcomes)
    assert not reserve._campaign_claim_path(reserve.C36_C60_CAMPAIGN).exists()
    assert (layout.claim_root / "CLAIM.json").is_file()


def test_default_c96_keeps_legacy_claim_without_source_hash_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_layout_roots(tmp_path, monkeypatch)
    targets = reserve_targets(("c96s1",))
    layout = reserve.execution_layout()
    monkeypatch.setattr(
        reserve, "_source_video_sha256",
        lambda _target: pytest.fail("default c96 claimでsource hashを読んではいけません"),
    )

    reserve._prepare_claim(layout, targets)

    fixed = reserve._fixed_campaign_claim_path()
    attempts = tuple((fixed / reserve.C96_CAMPAIGN_ATTEMPTS_DIR).glob("*.json"))
    assert len(attempts) == 1
    claim = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert claim["ordered_target_ids"] == ["c96s1"]
    assert "source_contract_sha256" not in claim


def test_default_c96_subset_keeps_legacy_queue_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = reserve_targets(("c96s1",))
    layout = replace(_versioned_campaign_layout(tmp_path), version=None)
    prepared: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        reserve, "_parse_args",
        lambda: SimpleNamespace(target=["c96s1"], artifact_version=None),
    )
    monkeypatch.setattr(reserve, "reserve_targets", lambda _requested: targets)
    monkeypatch.setattr(reserve, "execution_layout", lambda _version: layout)
    monkeypatch.setattr(
        reserve, "_prepare_claim",
        lambda _layout, selected: prepared.append(
            tuple(target.target_id for target in selected)
        ),
    )
    monkeypatch.setattr(
        reserve, "_process_target",
        lambda target, _layout: reserve.TargetOutcome(
            target.target_id, "audit_passed", True,
        ),
    )

    assert reserve.main() == 0
    assert prepared == [("c96s1",)]
