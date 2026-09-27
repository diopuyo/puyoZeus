"""Phase J左右ロゴterminal adapterの因果・allowlist契約試験。"""

from __future__ import annotations

import numpy as np
import pytest

from src.phase_j.reducer import ReducerEventKind
from src.phase_j.terminal_adapter import PhaseJTerminalAdapter, TerminalFrameContext
from src.review_terminal_outcome import ReviewTerminalEvidence


class _TwoHitDetector:
    def __init__(self, *, bilateral: bool = True) -> None:
        self.bilateral = bilateral
        self.hits = 0
        self.resets = 0

    def update_symmetric(self, _frame: np.ndarray) -> ReviewTerminalEvidence | None:
        self.hits += 1
        if self.hits < 2:
            return None
        return ReviewTerminalEvidence(
            "p2", "match_end_yatta+batan", 0.99, self.hits, self.bilateral,
        )

    def reset(self) -> None:
        self.hits = 0
        self.resets += 1


def _context(
    frame: int, *, match_id: str = "match-1", event_seq: int = 4,
    asset_bundle_id: str = "assets-1",
) -> TerminalFrameContext:
    return TerminalFrameContext(
        event_seq=event_seq, match_id=match_id, capture_session_id="capture-1",
        asset_bundle_id=asset_bundle_id, source_available_frame=frame,
        source_available_ms=frame * 33, captured_monotonic_ms=frame * 10,
        observed_monotonic_ms=frame * 10 + 1,
    )


def test_adapter_emits_only_bilateral_allowlisted_evidence() -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    unilateral = PhaseJTerminalAdapter(_TwoHitDetector(bilateral=False))
    assert unilateral.update(frame, _context(10)) is None
    assert unilateral.update(frame, _context(11)) is None

    adapter = PhaseJTerminalAdapter(_TwoHitDetector())
    assert adapter.update(frame, _context(10)) is None
    event = adapter.update(frame, _context(11))

    assert event is not None
    assert event.kind is ReducerEventKind.TERMINAL_EVIDENCE
    assert event.payload["winner"] == "1P"
    assert event.payload["result_code"] == "p1_win"
    assert event.payload["evidence_kind"] == "visual_result_logo_bilateral_2x2"
    assert event.content_digest.startswith("sha256:")
    assert adapter.update(frame, _context(12)) is None


def test_frame_gap_resets_consecutive_confirmation() -> None:
    detector = _TwoHitDetector()
    adapter = PhaseJTerminalAdapter(detector)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    assert adapter.update(frame, _context(10)) is None
    assert adapter.update(frame, _context(13)) is None
    event = adapter.update(frame, _context(14))

    assert event is not None
    assert detector.resets == 2


def test_match_change_resets_one_shot_latch() -> None:
    adapter = PhaseJTerminalAdapter(_TwoHitDetector())
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    assert adapter.update(frame, _context(10)) is None
    assert adapter.update(frame, _context(11)) is not None
    assert adapter.update(frame, _context(12)) is None

    assert adapter.update(frame, _context(1, match_id="match-2", event_seq=5)) is None
    event = adapter.update(frame, _context(2, match_id="match-2", event_seq=5))
    assert event is not None
    assert event.payload["match_id"] == "match-2"


def test_asset_bundle_change_resets_consecutive_confirmation() -> None:
    detector = _TwoHitDetector()
    adapter = PhaseJTerminalAdapter(detector)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    assert adapter.update(frame, _context(10)) is None
    assert adapter.update(
        frame, _context(11, asset_bundle_id="assets-2"),
    ) is None
    event = adapter.update(frame, _context(12, asset_bundle_id="assets-2"))

    assert event is not None
    assert event.payload["asset_bundle_id"] == "assets-2"


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_context_rejects_invalid_event_sequence(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TerminalFrameContext(
            value, "match", "capture", "assets", 1, 1, 1, 1,  # type: ignore[arg-type]
        )
