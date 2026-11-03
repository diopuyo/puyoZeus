"""左右勝敗ロゴの確定証拠だけをPhase J reducer eventへ変換する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.review_terminal_outcome import (
    ReviewTerminalEvidence,
    ReviewTerminalOutcomeDetector,
)

from .contracts import TerminalEvidenceKind, TerminalResultCode, TerminalWinner
from .reducer import ReducerEvent, ReducerEventKind


MAX_CONSECUTIVE_FRAME_GAP = 2
SYMMETRIC_TEMPLATE_NAME = "match_end_yatta+batan"


class BilateralTerminalDetector(Protocol):
    """左右ロゴの同時整合だけを返す検出器契約。"""

    def update_symmetric(self, frame: np.ndarray) -> ReviewTerminalEvidence | None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TerminalFrameContext:
    """一枚のsource frameを因果eventへ束縛する識別情報。"""

    event_seq: int
    match_id: str
    capture_session_id: str
    asset_bundle_id: str
    source_available_frame: int
    source_available_ms: int
    captured_monotonic_ms: int
    observed_monotonic_ms: int

    def __post_init__(self) -> None:
        integers = (
            self.event_seq, self.source_available_frame, self.source_available_ms,
            self.captured_monotonic_ms, self.observed_monotonic_ms,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in integers
        ):
            raise ValueError("terminal frameの通番・時刻は非負整数である必要があります")
        if not self.match_id or not self.capture_session_id or not self.asset_bundle_id:
            raise ValueError("terminal frameのidentityは非空である必要があります")


class PhaseJTerminalAdapter:
    """frame連続性を検査し、bilateral証拠だけを一度event化する。"""

    def __init__(self, detector: BilateralTerminalDetector) -> None:
        self._detector = detector
        self._identity: tuple[str, str, str] | None = None
        self._last_frame: int | None = None
        self._emitted = False

    @classmethod
    def load_default(cls) -> "PhaseJTerminalAdapter":
        return cls(ReviewTerminalOutcomeDetector.load_default(yatta_logo_only=True))

    def reset(self) -> None:
        self._detector.reset()
        self._identity = None
        self._last_frame = None
        self._emitted = False

    def update(
        self, frame: np.ndarray, context: TerminalFrameContext,
    ) -> ReducerEvent | None:
        identity = (
            context.match_id, context.capture_session_id, context.asset_bundle_id,
        )
        if identity != self._identity:
            self.reset()
            self._identity = identity
        if self._frame_gap_invalid(context.source_available_frame):
            self._detector.reset()
        self._last_frame = context.source_available_frame
        if self._emitted:
            return None
        evidence = self._detector.update_symmetric(frame)
        if not _is_bilateral_evidence(evidence):
            return None
        self._emitted = True
        return _build_event(context, evidence)

    def _frame_gap_invalid(self, current: int) -> bool:
        if self._last_frame is None:
            return False
        gap = current - self._last_frame
        return gap <= 0 or gap > MAX_CONSECUTIVE_FRAME_GAP


def _is_bilateral_evidence(evidence: ReviewTerminalEvidence | None) -> bool:
    return bool(
        evidence is not None
        and evidence.bilateral
        and evidence.template_name == SYMMETRIC_TEMPLATE_NAME
        and evidence.loser in {"p1", "p2"}
    )


def _build_event(
    context: TerminalFrameContext, evidence: ReviewTerminalEvidence,
) -> ReducerEvent:
    winner = TerminalWinner.PLAYER_2 if evidence.loser == "p1" else TerminalWinner.PLAYER_1
    result = (
        TerminalResultCode.PLAYER_1_WIN
        if winner is TerminalWinner.PLAYER_1
        else TerminalResultCode.PLAYER_2_WIN
    )
    payload = {
        "match_id": context.match_id,
        "capture_session_id": context.capture_session_id,
        "asset_bundle_id": context.asset_bundle_id,
        "source_available_frame": context.source_available_frame,
        "source_available_ms": context.source_available_ms,
        "captured_monotonic_ms": context.captured_monotonic_ms,
        "observed_monotonic_ms": context.observed_monotonic_ms,
        "evidence_kind": TerminalEvidenceKind.VISUAL_RESULT_LOGO_BILATERAL_2X2.value,
        "winner": winner.value,
        "result_code": result.value,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return ReducerEvent(
        ReducerEventKind.TERMINAL_EVIDENCE, context.event_seq,
        f"sha256:{digest}", payload,
    )


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return text.encode("utf-8")


__all__ = [
    "BilateralTerminalDetector",
    "MAX_CONSECUTIVE_FRAME_GAP",
    "PhaseJTerminalAdapter",
    "TerminalFrameContext",
]
