"""実会計の読取値を開始資格用に保持する。初期ゼロを開始証拠にしない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

SIDES = ('p1', 'p2')
FPS = 60
ACTIVITY = ('generated', 'offset_uncapped', 'dropped_uncapped', 'clamp_loss')


def pair(value: Any, prefix: str) -> list:
    return [getattr(value, prefix + '_' + side) for side in SIDES]


def capture(state: dict, frame: int) -> dict:
    """原observe直後の同一trackerを読み、observer保持値とも突合する。"""
    tracker, recorder = state['ojama_tracker'], state['accounting_recorder']
    seconds = state['t_sec']
    assert state['fi'] == frame and seconds == frame / FPS, 'start_trace_clock'
    gross = tracker.get_gross_counters(seconds)
    final = tracker.get_attack_finalization_counters(seconds)
    snapshot = tracker.get_snapshot(seconds)
    assert asdict(gross) == asdict(recorder._previous_gross), 'start_trace_gross'
    assert asdict(final) == asdict(recorder._previous_final), 'start_trace_final'
    pending = [snapshot.pending_p1_uncapped, snapshot.pending_p2_uncapped]
    assert pending == list(recorder._previous_pending), 'start_trace_pending'
    shared, result, pipe = state['shared_game'], state['result'], state['pipeline']
    unsettled = []
    for side in SIDES:
        account = getattr(tracker, '_' + side)
        unsettled.append(bool(account.chain_active or account.score_settle_pending
                              or getattr(pipe, '_active_chain_' + side[-1] + 'p') is not None))
    return dict(frame=frame, game=shared.game_idx, active=result.is_match_active,
        gates=[shared.multisignal_mode, shared.require_newmatch_evidence],
        states=[getattr(result, side).state.value for side in SIDES],
        scores=[getattr(result, side).score for side in SIDES],
        resets=pair(gross, 'boundary_resets'),
        activity={name: pair(gross, name) for name in ACTIVITY} | {'finalized': pair(final, 'finalized_count')},
        pending=pending, capped=[snapshot.pending_p1, snapshot.pending_p2],
        leftover=pair(snapshot, 'leftover'), unsettled=unsettled)
