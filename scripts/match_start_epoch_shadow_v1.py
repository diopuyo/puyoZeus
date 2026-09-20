"""検収済みvideo38の単一reset episodeだけで終了ロックの世代を揃える。

汎用reset修復でも新ゲーム自動認証器でもない。後日の実画像検収を根拠に
固定episodeへ介入する診断専用hookで、score-zero・会計・STABLE許可は不変。
runnerはREQUIRED_INPUT_SHA256を開始/終了guardへ加え、元動画も照合すること。
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import math
import sys
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / ".runtime_snapshots/event_first30_observed_context_v5_2026-08-30"
PIPELINE_PATH = SNAPSHOT / "src/recognition_pipeline.py"
DETECTOR_PATH = SNAPSHOT / "src/match_end_detector.py"
PIPELINE_SHA = "6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02"
DETECTOR_SHA = "9546be6b0a124fa8dc1f1716b70c8777ebf631f7095a58a169e45c852ea9ae1c"
REVIEW_FRAME, SOURCE_FPS, RESET_CALL_LINE = 32494, 60, 4868
REVIEW_TIME = REVIEW_FRAME / SOURCE_FPS
EPISODE_ID = "video38:reviewed-score-reset:frame32494"
VIDEO_PATH = ROOT / "data/frames/video_38.mp4"
VIDEO_SHA = "b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3"
REQUIRED_INPUT_SHA256 = {
    str(PIPELINE_PATH): PIPELINE_SHA, str(DETECTOR_PATH): DETECTOR_SHA,
    str(VIDEO_PATH): VIDEO_SHA,
    str(ROOT / "data/verify/video38_accounting_start_review_2026-09-07_v1/PHYSICAL_QA.md"):
        "7f6fa93386267f786165488e17e9a3b7f95b09bc4ad0816bb6c8a1186d18d822",
    str(ROOT / "data/verify/video38_start_gate_2026-09-07_v1/START_GATE_CAUSAL_QA.md"):
        "fb825bbf31341f6cfe4db7cbf6ed129af6f2e8cf285911f6d45a221a26846203",
}
RESET_GATE_VALUES = {
    "_match_end_locked_since": -1.0, "_post_match_lockdown_active": False,
    "_post_match_lockdown_prev_end_locked": False,
    "_post_match_lockdown_started_time": -1.0,
    "_post_match_lockdown_raw_active_since": -1.0,
}
GATE_FIELDS = ("_last_match_end_locked", *RESET_GATE_VALUES)


def _sha256(path: Path) -> str:
    """小さい凍結sourceだけを照合する。動画全SHAはrunnerのguard責務。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_runtime(collector: ModuleType) -> ModuleType:
    """現行src混成や別版への行番号適用を設置前に拒否する。"""
    cls = collector.RecognitionPipeline
    module = sys.modules[cls.__module__]
    if Path(module.__file__).resolve() != PIPELINE_PATH.resolve():
        raise RuntimeError("開始epoch hookは固定snapshot専用です")
    detector = sys.modules[module.MatchEndDetector.__module__]
    if Path(detector.__file__).resolve() != DETECTOR_PATH.resolve():
        raise RuntimeError("MatchEndDetectorの凍結系譜が違います")
    for path, expected in ((PIPELINE_PATH, PIPELINE_SHA), (DETECTOR_PATH, DETECTOR_SHA)):
        if _sha256(path) != expected:
            raise RuntimeError(f"開始epoch source SHA不一致: {path}")
    return module


def _snapshot(pipe: Any) -> dict[str, Any]:
    """既存属性だけを複製する。判定器を追加呼出しない。"""
    values = {name: getattr(pipe, name) for name in GATE_FIELDS}
    detector = pipe._match_end_detector
    values["detector_last_detected_t"] = None if detector is None else detector.last_detected_t
    return values


def _evidence(caller: FrameType | None, pipe: Any, rec: Any,
              match_start_sec: float | None, module: ModuleType) -> dict[str, Any]:
    """既存の実branch/localを読む。スコア境界判定を再評価しない。"""
    local = {} if caller is None else caller.f_locals
    return {
        "episode_id": EPISODE_ID, "source_video_sha256_required": VIDEO_SHA,
        "review_basis": "既存実画像/旧receiptによる単一episodeの事前登録介入",
        "general_new_game_certification": False, "accounting_basis_verified": False,
        "caller_source": None if caller is None else caller.f_code.co_filename,
        "caller_function": None if caller is None else caller.f_code.co_name,
        "caller_line": None if caller is None else caller.f_lineno,
        "same_pipeline": local.get("self") is pipe,
        "frame_idx": getattr(rec, "frame", None), "time_sec": getattr(rec, "time_sec", None),
        "caller_frame_idx": local.get("frame_idx"), "caller_time_sec": local.get("time_sec"),
        "match_start_sec": match_start_sec,
        "boundary_candidate": local.get("boundary_candidate"), "boundary_now": local.get("boundary_now"),
        "strict": getattr(pipe, "_enable_score_reset_strict", False),
        "full_clear": getattr(pipe, "_enable_match_start_full_clear", False),
        "streak": getattr(pipe, "_score_reset_boundary_streak", None),
        "required_streak": module.SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES,
        "already_latched": getattr(pipe, "_match_start_boundary_latched", True),
    }


def _eligible(evidence: dict[str, Any]) -> bool:
    """allowlistだけでも、OCR条件だけでも修復を許可しない。"""
    source = evidence["caller_source"]
    clocks = (evidence[name] for name in ("time_sec", "caller_time_sec", "match_start_sec"))
    return (
        source is not None and Path(source).resolve() == PIPELINE_PATH.resolve()
        and evidence["caller_function"] == "update" and evidence["caller_line"] == RESET_CALL_LINE
        and evidence["same_pipeline"] is True
        and evidence["frame_idx"] == evidence["caller_frame_idx"] == REVIEW_FRAME
        and all(isinstance(t, (int, float)) and not isinstance(t, bool)
                and math.isfinite(t) and t == REVIEW_TIME for t in clocks)
        and all(evidence[name] is True for name in ("strict", "full_clear", "boundary_candidate", "boundary_now"))
        and evidence["already_latched"] is False
        and type(evidence["streak"]) is int
        and evidence["streak"] >= evidence["required_streak"]
    )


def _restore_gate(pipe: Any, detector: Any, values: dict[str, Any]) -> None:
    """追加修復だけをrollbackする。元resetの変更を巻き戻したとは主張しない。"""
    detector._last_detected_t = values["detector_last_detected_t"]
    for name in GATE_FIELDS:
        setattr(pipe, name, values[name])


def _repair(pipe: Any, module: ModuleType, rec: Any, evidence: dict[str, Any],
            before_reset: dict[str, Any]) -> None:
    """正常完了した既存resetに不足する2属性だけを整合させる。"""
    detector, before = pipe._match_end_detector, _snapshot(pipe)
    if type(detector) is not module.MatchEndDetector:
        raise RuntimeError("凍結MatchEndDetectorの実instanceではありません")
    if any(before[name] != expected for name, expected in RESET_GATE_VALUES.items()):
        raise RuntimeError("既存resetのlatch初期化契約が成立していません")
    try:
        detector.reset()
        pipe._last_match_end_locked = False
        after = _snapshot(pipe)
        if after["detector_last_detected_t"] is not None:
            raise RuntimeError("既存detector.resetが終了世代を解除していません")
        rec.emit({"kind": "boundary_repair_mutation", "repair": "start_epoch", "side": None,
                  "before": before, "after": after,
                  "evidence": {**evidence, "before_original_reset": before_reset,
                               "source_sha256_verification": "runner開始終了guard必須"}})
    except BaseException:
        _restore_gate(pipe, detector, before)
        raise


class EpochRepair:
    """単一episode介入の回数だけを外側に保持する。会計stateは持たない。"""

    def __init__(self, rec: Any, module: ModuleType) -> None:
        self.rec, self.module, self.applied = rec, module, False

    def observation(self, evidence: dict[str, Any], reason: str) -> None:
        """拒否/例外を実修復と区別し、元resetの意味を変更しない。"""
        self.rec.emit({"kind": "boundary_repair_observation", "repair": "start_epoch",
                       "side": None, "reason": reason, "evidence": evidence})

    def wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """元resetを一度だけ呼ぶ。任意reset・単発OCR・他episodeには介入しない。"""
        @functools.wraps(original)
        def reset(pipe: Any, match_start_sec: float | None = None) -> Any:
            current = inspect.currentframe()
            try:
                evidence = _evidence(current.f_back, pipe, self.rec, match_start_sec, self.module)
            finally:
                del current
            eligible = _eligible(evidence) and not self.applied
            before = _snapshot(pipe) if eligible else None
            try:
                result = original(pipe, match_start_sec=match_start_sec)
            except BaseException:
                try:
                    self.observation(evidence, "original_reset_exception_no_added_repair")
                except BaseException:
                    pass  # 完了不能なrunの記録障害で、元resetの例外を隠さない。
                raise
            if not eligible:
                self.observation(evidence, "outside_reviewed_episode_or_already_applied")
                return result
            _repair(pipe, self.module, self.rec, evidence, before)
            self.applied = True
            return result
        return reset


def install(stack: contextlib.ExitStack, collector: ModuleType, rec: Any) -> EpochRepair:
    """fresh frozen collector後だけ設置し、成功/例外ともdescriptorを復元する。"""
    module = _validate_runtime(collector)
    cls = collector.RecognitionPipeline
    original = inspect.getattr_static(cls, "reset")
    repair = EpochRepair(rec, module)
    stack.callback(setattr, cls, "reset", original)
    setattr(cls, "reset", repair.wrap(cls.reset))
    return repair
