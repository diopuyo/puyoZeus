"""人工初期context/保存rawで、原SMの短い落下mergeを比較する。"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'g2_midmatch_side_resync_candidate_2026-09-11_v1'))
import check_candidate as Q
from src.board_state_machine import DetectorSignals, BoardState
import endpoint_votes as V


def objects() -> tuple[Any, Any]:
    source = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v32/LIVE_EMPTY_RESET.json'
    row = next(r for r in json.loads(source.read_bytes())['recovery'] if r.get('frame') == 34944)
    raw = Q.Board.from_dict({'grid': row['raw']})
    pipe = Q.repro.RecognitionPipeline(image_reader=Q.repro.MagicMock(spec=Q.repro.ImageReader),
        match_state_detector=Q.repro.MagicMock(spec=Q.repro.MatchStateDetector), stable_frame_count=2)
    sm = pipe._sm_1p
    sm.context.state = BoardState.STABLE
    sm.context.confirmed_board = Q.Board.from_dict(row['channels']['confirmed'])
    return sm, raw


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_sm_short_fall(enabled: bool) -> None:
    sm, raw = objects()
    with contextlib.ExitStack() as stack:
        value = V.install(stack, sm, lambda: enabled)
        for frame in range(34946, 34954, 2):
            result = sm.update(frame, DetectorSignals(frame / 60, raw.copy(), True,
                next_pair=(4, 5), hsv_board=raw.copy(), effect_gate_window_active=False))
        assert result.state == BoardState.STABLE
        assert [(result.confirmed_board.get(r, 4)) for r in (1, 2)] == ([4, 3] if enabled else [0, 0])
        assert len(value.rows) == int(enabled)
        if enabled:
            assert value.rows[0]['frames'] == [34948, 34950, 34952]
        # UNKNOWN行は原mergeが保持する。ここで隠し確率の解決は主張しない。
        assert result.confirmed_board.get(0, 4) == 0


@pytest.mark.parametrize('fault', ['duplicate', 'gap', 'effect', 'changed', 'foreign_history', 'chain'])
def test_reject_invalid_observations(fault: str) -> None:
    _, raw = objects()
    ticks = [V.Tick(0, V.STABLE, V.FALL, raw.copy(), True),
             V.Tick(2, V.FALL, V.FALL, raw.copy(), True),
             V.Tick(4, V.FALL, None, raw.copy(), True)]
    history = [raw.copy()]
    if fault == 'duplicate': ticks[1].frame = 0
    if fault == 'gap': ticks[1].frame = 3
    if fault == 'effect': ticks[1].quiet = False
    if fault == 'changed': ticks[1].board.set(2, 4, 2)
    if fault == 'foreign_history': history[0].set(2, 4, 2)
    if fault == 'chain': ticks[0].before = 'chain'
    assert V.extended_history(ticks, history) is None
