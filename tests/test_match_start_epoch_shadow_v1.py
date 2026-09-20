"""開始epochの限定介入・一般reset互換・例外復元をCPUだけで確認する。"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import match_start_epoch_shadow_v1 as shadow


class Detector:
    """タイマー以外へ触れない既存detectorの小さい代役。"""

    def __init__(self) -> None:
        self._last_detected_t = 541.2
        self.calls = 0
        self.failure = False

    @property
    def last_detected_t(self) -> float | None:
        return self._last_detected_t

    def reset(self) -> None:
        self.calls += 1
        self._last_detected_t = None
        if self.failure:
            raise RuntimeError("detector fixture")


class Pipeline:
    """元resetの会計クリアと戻り値を維持するCPU代役。"""

    def __init__(self) -> None:
        self._match_end_detector = Detector()
        self._last_match_end_locked = True
        for name in shadow.RESET_GATE_VALUES:
            setattr(self, name, True if "active" in name or "locked" in name else 541.2)
        self._enable_score_reset_strict = True
        self._enable_match_start_full_clear = True
        self._score_reset_boundary_streak = 3
        self._match_start_boundary_latched = False
        self.calls = 0
        self.counter = {4: 2}
        self.failure = None
        self.error = RuntimeError("original fixture")

    def reset(self, match_start_sec: float | None = None) -> str:
        self.calls += 1
        self.match_start_sec = match_start_sec
        self.counter.clear()
        if self.failure == "original":
            raise self.error
        for name, expected in shadow.RESET_GATE_VALUES.items():
            setattr(self, name, expected)
        if self.failure == "latch":
            self._post_match_lockdown_active = True
        return "unchanged-return"


def module() -> Any:
    return SimpleNamespace(MatchEndDetector=Detector, SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES=3)


def recorder() -> Any:
    rec = SimpleNamespace(frame=shadow.REVIEW_FRAME, time_sec=shadow.REVIEW_TIME, rows=[])
    rec.emit = rec.rows.append
    return rec


def evidence() -> dict[str, Any]:
    """実caller/local条件の全てを指定し、欠測を認証へ補完しない。"""
    return {
        "caller_source": str(shadow.PIPELINE_PATH), "caller_function": "update",
        "caller_line": shadow.RESET_CALL_LINE, "same_pipeline": True,
        "frame_idx": shadow.REVIEW_FRAME, "caller_frame_idx": shadow.REVIEW_FRAME,
        "time_sec": shadow.REVIEW_TIME, "caller_time_sec": shadow.REVIEW_TIME,
        "match_start_sec": shadow.REVIEW_TIME, "strict": True, "full_clear": True,
        "boundary_candidate": True, "boundary_now": True, "already_latched": False,
        "streak": 3, "required_streak": 3,
    }


@pytest.mark.parametrize("key,value", [
    ("caller_source", None), ("caller_source", "src/recognition_pipeline.py"),
    ("caller_function", "reset"), ("caller_line", 4867), ("same_pipeline", False),
    ("frame_idx", 32492), ("caller_frame_idx", 32492), ("time_sec", 541.5),
    ("caller_time_sec", 541.5), ("match_start_sec", None), ("match_start_sec", float("nan")),
    ("match_start_sec", float("inf")), ("match_start_sec", True),
    ("strict", False), ("full_clear", False), ("boundary_candidate", False),
    ("boundary_now", False), ("boundary_now", None), ("already_latched", True),
    ("streak", 1), ("streak", 2), ("streak", True), ("streak", None),
])
def test_incomplete_or_other_episode_is_not_eligible(key: str, value: Any) -> None:
    values = {**evidence(), key: value}
    assert not shadow._eligible(values)


def test_complete_existing_branch_and_allowlist_are_both_required() -> None:
    assert shadow._eligible(evidence())


def test_evidence_reads_actual_locals_without_recomputing_boundary() -> None:
    pipe, rec = Pipeline(), recorder()
    caller = SimpleNamespace(f_locals={"self": pipe, "frame_idx": rec.frame,
        "time_sec": rec.time_sec, "boundary_candidate": True, "boundary_now": True},
        f_code=SimpleNamespace(co_filename=str(shadow.PIPELINE_PATH), co_name="update"),
        f_lineno=shadow.RESET_CALL_LINE)
    values = shadow._evidence(caller, pipe, rec, rec.time_sec, module())
    assert shadow._eligible(values)
    assert values["general_new_game_certification"] is False
    assert values["accounting_basis_verified"] is False
    assert pipe._match_end_detector.calls == 0


def test_missing_caller_is_not_promoted() -> None:
    values = shadow._evidence(None, Pipeline(), recorder(), shadow.REVIEW_TIME, module())
    assert not shadow._eligible(values)


def hook(monkeypatch: Any, pipe: Pipeline, rec: Any) -> tuple[Any, Any]:
    """unitではcallerを固定し、fresh subprocessで実callerも別途検査する。"""
    monkeypatch.setattr(shadow, "_evidence", lambda *args: evidence())
    repair = shadow.EpochRepair(rec, module())
    return repair, repair.wrap(Pipeline.reset)


def test_repair_fills_only_detector_and_cache_after_original_reset(monkeypatch: Any) -> None:
    pipe, rec = Pipeline(), recorder()
    repair, reset = hook(monkeypatch, pipe, rec)
    assert reset(pipe, shadow.REVIEW_TIME) == "unchanged-return"
    assert pipe.calls == pipe._match_end_detector.calls == 1 and repair.applied
    assert pipe.counter == {} and pipe.match_start_sec == shadow.REVIEW_TIME
    assert pipe._match_end_detector.last_detected_t is None
    assert pipe._last_match_end_locked is False
    row = rec.rows[0]
    assert row["kind"] == "boundary_repair_mutation" and row["side"] is None
    changed = {key for key in row["before"] if row["before"][key] != row["after"][key]}
    assert changed == {"_last_match_end_locked", "detector_last_detected_t"}
    assert row["evidence"]["before_original_reset"]["_post_match_lockdown_active"]


def test_duplicate_episode_does_not_apply_twice(monkeypatch: Any) -> None:
    pipe, rec = Pipeline(), recorder()
    repair, reset = hook(monkeypatch, pipe, rec)
    reset(pipe, shadow.REVIEW_TIME)
    reset(pipe, shadow.REVIEW_TIME)
    assert pipe.calls == 2 and pipe._match_end_detector.calls == 1
    assert repair.applied and rec.rows[1]["kind"] == "boundary_repair_observation"


@pytest.mark.parametrize("clock", [None, 0.0, shadow.REVIEW_TIME, 600.0])
def test_external_resets_preserve_detector_even_with_start_time(clock: float | None) -> None:
    pipe, rec = Pipeline(), recorder()
    repair = shadow.EpochRepair(rec, module())
    assert repair.wrap(Pipeline.reset)(pipe, clock) == "unchanged-return"
    assert pipe.calls == 1 and pipe._match_end_detector.calls == 0
    assert pipe._match_end_detector.last_detected_t == 541.2
    assert pipe._last_match_end_locked is True and not repair.applied


def test_original_reset_exception_has_no_added_repair(monkeypatch: Any) -> None:
    pipe, rec = Pipeline(), recorder()
    pipe.failure = "original"
    repair, reset = hook(monkeypatch, pipe, rec)
    with pytest.raises(RuntimeError) as caught:
        reset(pipe, shadow.REVIEW_TIME)
    assert caught.value is pipe.error and not repair.applied
    assert pipe.calls == 1 and pipe._match_end_detector.calls == 0
    assert rec.rows[0]["reason"] == "original_reset_exception_no_added_repair"


def test_receipt_failure_does_not_mask_original_exception(monkeypatch: Any) -> None:
    pipe, rec = Pipeline(), recorder()
    pipe.failure = "original"
    def fail_emit(row: Any) -> None:
        raise ValueError("recording failure")
    rec.emit = fail_emit
    repair, reset = hook(monkeypatch, pipe, rec)
    with pytest.raises(RuntimeError) as caught:
        reset(pipe, shadow.REVIEW_TIME)
    assert caught.value is pipe.error and not repair.applied
    assert pipe._match_end_detector.calls == 0


@pytest.mark.parametrize("failure", ["detector", "emit", "latch", "wrong_type"])
def test_added_repair_failure_restores_only_its_gate(monkeypatch: Any, failure: str) -> None:
    pipe, rec = Pipeline(), recorder()
    repair, reset = hook(monkeypatch, pipe, rec)
    if failure == "detector": pipe._match_end_detector.failure = True
    if failure == "latch": pipe.failure = "latch"
    if failure == "wrong_type": pipe._match_end_detector = SimpleNamespace(last_detected_t=541.2)
    if failure == "emit":
        def fail_emit(row: Any) -> None:
            raise RuntimeError("emit fixture")
        rec.emit = fail_emit
    with pytest.raises(RuntimeError): reset(pipe, shadow.REVIEW_TIME)
    assert not repair.applied and pipe.counter == {} and pipe.calls == 1
    assert pipe._match_end_detector.last_detected_t == 541.2
    assert pipe._last_match_end_locked is True


@pytest.mark.parametrize("failure", [False, True])
def test_install_restores_exact_descriptor(monkeypatch: Any, failure: bool) -> None:
    monkeypatch.setattr(shadow, "_validate_runtime", lambda collector: module())
    original = inspect.getattr_static(Pipeline, "reset")
    try:
        with contextlib.ExitStack() as stack:
            shadow.install(stack, SimpleNamespace(RecognitionPipeline=Pipeline), recorder())
            assert inspect.getattr_static(Pipeline, "reset") is not original
            if failure: raise RuntimeError("scope fixture")
    except RuntimeError:
        assert failure
    assert inspect.getattr_static(Pipeline, "reset") is original


def test_runtime_rejects_current_source_before_patching() -> None:
    original = Pipeline.reset
    with contextlib.ExitStack() as stack, pytest.raises(RuntimeError, match="snapshot"):
        shadow.install(stack, SimpleNamespace(RecognitionPipeline=Pipeline), recorder())
    assert Pipeline.reset is original


@pytest.mark.parametrize("failure", [None, "pipeline_path", "detector_path", "pipeline_sha", "detector_sha"])
def test_runtime_path_and_sha_both_required(monkeypatch: Any, failure: str | None) -> None:
    detector = SimpleNamespace(__module__="_epoch_test_detector")
    pipeline = SimpleNamespace(__file__=str(shadow.PIPELINE_PATH), MatchEndDetector=detector)
    detector_module = SimpleNamespace(__file__=str(shadow.DETECTOR_PATH))
    cls = SimpleNamespace(__module__="_epoch_test_pipeline")
    monkeypatch.setitem(sys.modules, cls.__module__, pipeline)
    monkeypatch.setitem(sys.modules, detector.__module__, detector_module)
    if failure == "pipeline_path": pipeline.__file__ = "src/recognition_pipeline.py"
    if failure == "detector_path": detector_module.__file__ = "src/match_end_detector.py"
    digests = {shadow.PIPELINE_PATH: shadow.PIPELINE_SHA, shadow.DETECTOR_PATH: shadow.DETECTOR_SHA}
    if failure == "pipeline_sha": digests[shadow.PIPELINE_PATH] = "wrong"
    if failure == "detector_sha": digests[shadow.DETECTOR_PATH] = "wrong"
    monkeypatch.setattr(shadow, "_sha256", lambda path: digests[path])
    if failure is None:
        assert shadow._validate_runtime(SimpleNamespace(RecognitionPipeline=cls)) is pipeline
    else:
        with pytest.raises(RuntimeError):
            shadow._validate_runtime(SimpleNamespace(RecognitionPipeline=cls))


FRESH_PROBE = r'''
import contextlib, importlib.util, inspect, json, sys
from pathlib import Path
from types import SimpleNamespace
from scripts import match_start_epoch_shadow_v1 as shadow
from scripts import diagnose_video38_confirmed_collapse_v1 as base
collector = base.load_collector()
root = shadow.ROOT
spec = importlib.util.spec_from_file_location("_epoch_w38_fixture", root / "tests/test_w38_match_start_wiring_2026-08-25.py")
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)
from src.match_end_detector import MatchEndDetector, MatchEndDetectionResult
from src.recognition_pipeline import RecognitionPipeline
import torch
original = inspect.getattr_static(RecognitionPipeline, "reset")
paths = (shadow.PIPELINE_PATH, shadow.DETECTOR_PATH)
before = {str(p): shadow._sha256(p) for p in paths}
rec = SimpleNamespace(frame=0, time_sec=0.0, rows=[])
rec.emit = rec.rows.append
pipe = fixture._make_pipe_with_trackers()
pipe._enable_match_start_full_clear = True
pipe._enable_score_reset_strict = True
pipe._enable_post_match_lockdown_latch = True
pipe._match_end_detector = MatchEndDetector({})
pipe._match_end_detector._last_detected_t = 541.2
pipe._last_match_end_locked = True
with contextlib.ExitStack() as stack:
    repair = shadow.install(stack, collector, rec)
    for frame in (32490, 32492, 32494, 32496):
        rec.frame, rec.time_sec = frame, frame / 60
        pipe.update(frame, frame / 60, fixture._dummy_frame())
    assert repair.applied
    assert not pipe._last_match_end_locked and not pipe._post_match_lockdown_active
    assert pipe._match_end_detector.last_detected_t is None
    assert len([r for r in rec.rows if r["kind"] == "boundary_repair_mutation"]) == 1
    detector = pipe._match_end_detector
    detector.detect = lambda frame: MatchEndDetectionResult(True, "fixture", 1.0)
    assert detector.update(None, 550.0) and detector.last_detected_t == 550.0
    pipe._last_match_end_locked = True
    pipe._update_post_match_lockdown_latch(None, 550.0, True, False, 100, 100, False)
    assert pipe._post_match_lockdown_active
    pipe.reset(match_start_sec=560.0)
    assert detector.last_detected_t == 550.0 and pipe._last_match_end_locked
assert inspect.getattr_static(RecognitionPipeline, "reset") is original
assert before == {str(p): shadow._sha256(p) for p in paths}
assert not torch.cuda.is_initialized()
assert all(Path(m.__file__).resolve().is_relative_to(base.SNAPSHOT.resolve())
           for name, m in sys.modules.items() if name == "src" or name.startswith("src."))
print(json.dumps({"real_frozen_reset": True, "old_lock_rearmed": False,
                  "new_end_rearmed": True, "general_reset_unchanged": True,
                  "restored": True, "source_sha_unchanged": before, "cuda_initialized": False}))
'''


def test_fresh_frozen_actual_boundary_and_detector_without_cuda() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-B", "-c", FRESH_PROBE], cwd=shadow.ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["real_frozen_reset"] and report["new_end_rearmed"]
    assert not report["old_lock_rearmed"] and not report["cuda_initialized"]
