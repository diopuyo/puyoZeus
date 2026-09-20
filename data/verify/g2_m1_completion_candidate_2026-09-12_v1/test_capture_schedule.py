"""採録要求・保存完了・有限終端を区別する。実資格判定/保存器は別の結合試験。"""
from __future__ import annotations

from typing import Any
import pytest
import capture_schedule as S


@pytest.mark.parametrize('ready', [35370, 35450])
def test_first_qualified_requests_with_fixed_gap(ready: int) -> None:
    state, requested, sentinels = S.Schedule(), [], []
    for frame in range(35368, S.END + 1, S.STRIDE):
        state, row = S.observe(state, frame, ('2P:not_stable',) if frame < ready else ())
        if row['original_sentinel']: sentinels.append(row)
        if row['action'] == 'REQUEST':
            assert len(state.accepted) == len(requested)
            requested.append(frame)
            state = S.accept(state, frame)
    assert S.finish(state) == (ready, ready + S.MIN_GAP)
    assert [r['frame'] for r in sentinels] == list(S.EARLIEST)
    if ready > S.EARLIEST[0]: assert all(r['action'] == 'WAIT' for r in sentinels)


def test_deadline_without_qualification_fails() -> None:
    state = S.Schedule()
    for frame in range(35368, S.END + 1, S.STRIDE):
        state, row = S.observe(state, frame, ('2P:not_stable',))
        assert row['action'] == 'WAIT'
    with pytest.raises(ValueError, match='incomplete_coverage'): S.finish(state)


def test_pending_save_cannot_be_skipped_or_counted() -> None:
    state, _ = S.observe(S.Schedule(), 35368, ())
    state, _ = S.observe(state, 35370, ())
    assert state.accepted == ()
    with pytest.raises(ValueError, match='previous_save_unfinished'): S.observe(state, 35372, ())
    with pytest.raises(ValueError, match='saved_frame'): S.accept(state, 35372)
    following = S.accept(state, 35370)
    with pytest.raises(ValueError, match='saved_frame'): S.accept(following, 35370)


@pytest.mark.parametrize('frame', [35370, 35369, True, 36299])
def test_bad_first_or_odd_clock(frame: Any) -> None:
    with pytest.raises(ValueError): S.observe(S.Schedule(), frame, ())


@pytest.mark.parametrize('frame', [35368, 35372, 35366])
def test_no_duplicate_skip_or_reverse(frame: int) -> None:
    state, _ = S.observe(S.Schedule(), 35368, ())
    with pytest.raises(ValueError): S.observe(state, frame, ())


@pytest.mark.parametrize('reasons', [None, ['held'], ('',), ('held', 'held'), (1,)])
def test_invalid_hold_representation(reasons: Any) -> None:
    with pytest.raises(ValueError, match='reasons'): S.observe(S.Schedule(), 35368, reasons)


def test_partial_run_is_not_completion() -> None:
    state = S.Schedule(last=35410, accepted=(35370, 35410))
    with pytest.raises(ValueError, match='incomplete_coverage'): S.finish(state)
