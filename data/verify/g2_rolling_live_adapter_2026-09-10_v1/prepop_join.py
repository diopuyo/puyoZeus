"""実call前チェックと保存されたrolling消費の同一手を照合する。"""
from __future__ import annotations
from typing import Any

FIELDS = frozenset(('frame', 'side', 'token', 'evidence_id', 'queue_slots',
    'state_action', 'current_permission'))
PAIR_SIZE = 2


def check(checks: list[Any], linked: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
    links = linked['rolling_links']
    assert type(checks) is list and len(checks) == len(links), 'rolling_prepop_coverage'
    by_key = {(r['scope']['frame_idx'],r['scope']['side']):r for r in rows}
    link_keys = {(r['frame'],r['scope']['side']):r for r in links}
    assert len(link_keys) == len(links), 'rolling_link_duplicate'
    seen = set()
    for receipt in checks:
        assert type(receipt) is dict and set(receipt) == FIELDS, 'rolling_prepop_fields'
        assert type(receipt['frame']) is int and type(receipt['side']) is str, 'rolling_prepop_types'
        key = receipt['frame'], receipt['side']
        assert key not in seen and key in link_keys, 'rolling_prepop_scope_duplicate'
        seen.add(key)
        link = link_keys[key]
        assert key == (link['frame'],link['scope']['side']), 'rolling_prepop_scope'
        assert receipt['token'] == link['token'] and receipt['evidence_id'] == link['event_id'], 'rolling_prepop_event'
        assert type(receipt['queue_slots']) is int and receipt['queue_slots'] == PAIR_SIZE
        assert receipt['current_permission'] is False, 'rolling_prepop_authority'
        history = by_key[key]['decision']['history_state']['history']
        entries = [e for e in history if e['event_id'] == receipt['evidence_id']]
        assert len(entries) == 1 and type(receipt['state_action']) is int
        assert entries[0]['action'] == receipt['state_action'], 'rolling_prepop_action'
    return dict(prepop_saved_verified=True, rolling_exercised=bool(checks), count=len(checks),
        physical_certified=False, current_permission=False, quality_gate_clear=False)
