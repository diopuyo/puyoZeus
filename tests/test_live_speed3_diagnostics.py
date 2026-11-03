"""保存診断の欠測扱いと、証明を除外しない比較を検証する。"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import summarize_live_speed3 as summary
from scripts.diagnose_live_speed3 import event_costs, group
from scripts.compare_live_speed3_candidate import simulation
from scripts.summarize_live_speed3 import without_cost


def test_missing_child_timing_is_unknown(tmp_path: Path) -> None:
    assert event_costs(tmp_path/'missing.jsonl') == ({}, {})


def test_multiple_requests_per_frame_are_accumulated(tmp_path: Path) -> None:
    path = tmp_path/'events.jsonl'
    rows = [dict(frame=1, stages={name: dict(wall_ms=value)}) for name, value in
            (('event_update', 20), ('event_reference', 3), ('event_decode', 2))]
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    costs, totals = event_costs(path)
    assert costs == {1: 23}
    assert totals['event_decode'] == 2


def test_chain_and_ojama_are_not_double_counted() -> None:
    assert group(dict(state1='CHAIN', state2='OJAMA_FALL')) == 'chain'
    assert group(dict(state1='STABLE', state2='OJAMA_FALL')) == 'ojama'


def test_comparison_preserves_death_proof_and_video_time() -> None:
    value = dict(elapsed_sec=1, rows=[dict(t_sec=2, enumeration_sec=3,
                                         proof=dict(dead=True, filter_sec=4))])
    assert without_cost(value) == dict(rows=[dict(t_sec=2, proof=dict(dead=True))])


@pytest.mark.parametrize('changed', ['proof', 'display'])
def test_comparison_rejects_changed_judgment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           changed: str) -> None:
    reference, actual = tmp_path/'reference', tmp_path/'actual'
    for directory in (reference, actual):
        directory.mkdir()
        for name in ('events.jsonl', 'history_integrity_audit.json', *summary.AUDITS):
            (directory/name).write_text('{"dead": true}', encoding='utf-8')
        np.savez(directory/'display.npz', probability=np.array([0.5]))
    monkeypatch.setattr(summary, 'REFERENCE', reference)
    assert summary.compare(actual)['events.jsonl'] == 'byte_identical'
    if changed == 'proof':
        (actual/'hidden_death_audit.json').write_text('{"dead": false}', encoding='utf-8')
    else:
        np.savez(actual/'display.npz', probability=np.array([np.nextafter(0.5, 1)]))
    with pytest.raises(AssertionError):
        summary.compare(actual)


def test_queue_estimate_keeps_regressions_and_missing_samples() -> None:
    profiles = [dict(frame=1, recognized_at=1., started_at=1., finished_at=2., evaluation_wall_sec=1.),
                dict(frame=2, recognized_at=1.5, started_at=2., finished_at=3., evaluation_wall_sec=1.)]
    capture = {1: 1., 2: 1.5}
    baseline = simulation(profiles, capture, {})
    slower = simulation(profiles, capture, {1: -0.5})
    assert baseline['latency_ms']['mean'] == 1250
    assert slower['latency_ms']['mean'] == 1750
    # 短縮値のない2件目を削除したり、改善側だけを採用したりしない。
    assert slower['latency_ms']['n'] == 2
