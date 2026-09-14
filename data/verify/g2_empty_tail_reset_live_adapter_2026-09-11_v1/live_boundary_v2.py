"""旧公開も実旧owner scope・current証拠・consumerへ結び、空対照と区別する。"""
from __future__ import annotations
from typing import Any
import live_boundary as B


def check(consumer: Any, history: Any, journal: Any, recovery: Any, *, old_owner_scope: Any) -> Any:
    result = B.check(consumer,history,journal,recovery)
    reset_frame = min(r['frame'] for r in recovery if r['kind']=='reset_baseline_wait')
    prior = [r for r in history if 'decision' in r and r['scope']['frame_idx']<reset_frame]
    issued = []
    for row in prior:
        state = row['decision']['history_state']
        assert state['scope']==old_owner_scope,'prior_actual_owner_scope'
        if row['decision']['current_permission']:
            assert row['decision']['current_proof'] is not None,'prior_current_proof'
            assert state['current'] is not None and state['current']['available'] is True,'prior_available_current'
            issued.append(row['scope']['frame_idx'])
    observed = [r['frame_idx'] for r in consumer if r['frame_idx']<reset_frame for _ in range(r['tickets_this_update'])]
    assert issued==observed==result['prior_issued_frames'],'prior_publication_join'
    return result|dict(prior_integer_publications_retained=bool(issued),
        prior_publication_join_verified=True,prior_publication_count=len(issued),prior_scope_rows=len(prior))
