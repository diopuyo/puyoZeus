"""Stage1のscope表を、自己申告行だけでなく実owner/基準時刻/reset資格へ結ぶ。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

STRIDE, FPS, SIDE = 2, 60, '1P'


def verify(lease: Any, factory: Any, baseline: Any, full: Any) -> None:
    recovery, proof = lease.recovery,lease.empty_evidence
    assert recovery.factory is proof.factory is factory and recovery.control is factory.controller
    new = factory.controller.history[SIDE]
    assert new is lease.empty_new_binding and new.owner is lease.empty_new_owner
    assert new.scope==lease.new_scope==recovery.evidence.scope(factory,recovery.pipe), 'new_full_scope'
    old, old_state, record = recovery.archive[0]
    assert len(recovery.archive)==1 and old is lease.archive.binding and old.owner.state is old_state
    assert asdict(old_state)==record['owner'], 'old_owner_changed'
    assert baseline['state']['scope']==asdict(new.owner.state.scope), 'baseline_actual_owner_scope'
    assert asdict(new.owner.state.baseline_through)==baseline['state']['baseline_through'], 'baseline_actual_clock'
    assert type(baseline['frame']) is int and baseline['frame']==new.owner.state.baseline_through.frame
    assert baseline['first_owned_frame']==baseline['frame']+STRIDE, 'baseline_first_owned'
    qualified = [r for r in lease.events if r['kind']=='reset_qualified']
    assert len(qualified)==1 and qualified[0]['frame']==proof.frame+STRIDE, 'reset_qualified_frame'
    assert qualified[0]['clock']==qualified[0]['frame']/FPS
    waits = [r for r in full if r.get('kind')=='reset_baseline_wait']
    assert [r['frame'] for r in waits]==list(range(proof.frame+STRIDE,baseline['frame']+STRIDE,STRIDE)), 'reset_wait_window'
    assert waits and waits[-1]['baseline'] is True
    assert all(tuple(r['scope'])==new.scope for r in waits), 'waiting_actual_scope'
    assert baseline['proof']['reset_source']['empty_retirement']==lease.retirement, 'baseline_retirement'
    for row in full:
        if 'decision' not in row: continue
        frame = row['scope']['frame_idx']
        owner = old_state if frame<=proof.frame else new.owner.state
        assert frame<=proof.frame or frame>=baseline['first_owned_frame'], 'unowned_window_history'
        assert row['decision']['history_state']['scope']==asdict(owner.scope), 'stage_actual_owner_scope'
