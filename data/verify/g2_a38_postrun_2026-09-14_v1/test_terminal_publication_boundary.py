"""人工の整数境界＋確率追跡＋終了尾部。元boundary.checkを通すCPU接続対照。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace as N
import pytest
import terminal_publication_boundary as C

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_live_probability_context_2026-09-12_v1'))
import probability_boundary as P


def fixture() -> tuple:
    scope = ('source', 'run', 3, 4, 5, 3, '1P')
    def observed(frame: int) -> dict:
        return dict(frame_idx=frame, source_id=scope[0], run_id=scope[1], pipe_object_id=scope[3],
            side='1P', generation=dict(reset_epoch=3))
    history = [dict(scope=observed(0), decision=dict(current_permission=False))]
    waiting = dict(kind='reset_baseline_wait', frame=2, scope=scope, baseline=False,
        reason='waiting', journal_token='step:2')
    history.append(waiting)
    tracking = [dict(scope=observed(4), journal_token='step:4',
        integer_current_published=False, quality_gate_clear=False)]
    journal = [dict(observed(frame), kind='step', token=f'step:{frame}', status='returned',
        exception=None, software_reset=3, events=[]) for frame in (2, 4)]
    consumer = [dict(frame_idx=frame, same_result_identity=True, comparison_completed=True,
        changed_sides=[], full_before={'x':1}, full_after={'x':1},
        tickets_this_update=0, released_this_update=0) for frame in range(0, 10, 2)]
    mode = N(arrival_ledger=N(clock=4, deadline=8), native=N(last_frame=4),
        arrival_capture=N(terminal_receipt={'synthetic':True}, terminal_last=8, terminal_rows=2, pending=None))
    args = dict(history_first=0, end_frame=4, proof_frame=0, basis_frame=2, scope=scope)
    return mode, consumer, history, journal, [waiting], tracking, args


def test_original_prefix_then_full_terminal_consumer() -> None:
    mode, *values, args = fixture()
    with pytest.raises(AssertionError):
        P.check(*values, **args)  # 元検査単独はNative凍結後の全consumerを拒否する。
    report = C.check(P, mode, *values, **args)
    assert report['tracking_updates'] == 1 and report['terminal_publication_updates'] == 2
    assert report['terminal_end'] == 8 and not report['terminal_original_journal_verified']


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'ticket', 'released', 'prefix', 'end'])
def test_reject_changed_or_missing_publication(mutation: str) -> None:
    mode, consumer, history, journal, recovery, tracking, args = fixture()
    if mutation == 'missing': consumer.pop()
    elif mutation == 'changed': consumer[-1]['same_result_identity'] = False
    elif mutation == 'ticket': consumer[-1]['tickets_this_update'] = 1
    elif mutation == 'released': consumer[-1]['released_this_update'] = 1
    elif mutation == 'prefix': tracking.clear()
    elif mutation == 'end': mode.arrival_capture.terminal_last = 6
    with pytest.raises((AssertionError, ValueError)):
        C.check(P, mode, consumer, history, journal, recovery, tracking, **args)
