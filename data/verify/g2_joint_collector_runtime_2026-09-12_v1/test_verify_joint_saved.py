"""既存の実子保存票を再用し、候補・完了・解除条件の陰性を確認する。"""
from __future__ import annotations
from pathlib import Path
import pytest
import verify_joint_saved as V

SAVED = V.ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1' / 'parent_saved_v1'


def test_original_two_saved_packets() -> None:
    result = V.packets(SAVED, 'events.jsonl', ('candidate0.json', 'candidate1.json'), (144, 146))
    assert [p['result']['frame'] for p in result] == [144, 146]


def test_completion_is_required_before_candidates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        V.packets(tmp_path, 'events.jsonl', ('candidate0.json', 'candidate1.json'), (144, 146))


@pytest.mark.parametrize('fault', ['observation', 'producer', 'recapture', 'completion_path'])
def test_candidate_corruption(fault: str) -> None:
    packet = V.read(SAVED / 'candidate0.json')
    ticket = V.read(SAVED / 'events.jsonl.request0.json')
    if fault == 'observation': packet['observation']['frame'] += 2
    elif fault == 'producer': packet['producer_sha256'] = '0' * 64
    elif fault == 'recapture': packet['parent_recapture_verified'] = False
    else: packet['session_completion_required'] = 'wrong.complete.json'
    with pytest.raises(ValueError):
        V.candidate(packet, ticket, SAVED / 'events.jsonl')


@pytest.mark.parametrize('fault', [None, 'cleanup', 'saved', 'error', 'source'])
def test_session_cleanup_contract(fault: str | None) -> None:
    session = dict(restored=True, observer_closed=True, witness_closed=True, error=None,
                   session_error=None, observer_error=None, witness_error=None,
                   saved=[dict(frame=f) for f in V.FRAMES])
    attach = dict(installed=True, restored=True, pipeline_update_unchanged=True, error=None, body_error=None)
    source = dict(original_exit=0, unchanged=True, entry_restored=True)
    if fault == 'cleanup': session['restored'] = False
    elif fault == 'saved': session['saved'].pop()
    elif fault == 'error': attach['body_error'] = 'failure'
    elif fault == 'source': source['unchanged'] = False
    if fault is None:
        V.lifecycle(session, attach, source)
    else:
        with pytest.raises(ValueError):
            V.lifecycle(session, attach, source)
