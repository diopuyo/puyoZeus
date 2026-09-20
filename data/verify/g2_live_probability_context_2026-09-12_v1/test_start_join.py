"""人工開始traceを原canonical保存まで接続する。実動画の欠測補完ではない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_producer_canonical_join_2026-09-12_v1'))
import start_join as S
import start_qualification as Q

RESET = 2
QUALIFIED = 90


def original() -> dict:
    path = ROOT.parent / 'g2_observed_start_anchor_2026-09-12_v1/positive_full_v3/START_CAPTURE.json'
    return json.loads(path.read_bytes())


def artificial(value: dict | None = None) -> dict:
    value = original() if value is None else deepcopy(value)
    account = value['accounting']
    reset = deepcopy(account['rows'][0])
    reset.update(frame_idx=RESET, formal_boundary=False)
    for side in S.SIDES:
        reset['deltas']['boundary_resets_' + side] = 1
        account['final_gross_counters']['boundary_resets_' + side] = 1
    account['rows'].insert(0, reset)
    account['nonzero_row_count'] += 1
    for row in value['stable_snapshots']:
        if row['frame_idx'] == QUALIFIED:
            row['arguments']['score'] = 6
    value['start_counter_trace'] = trace(value)
    value['start_parameters'] = dict(debounce_sec=Q.DEBOUNCE_SEC, visual_persist_sec=Q.VISUAL_PERSIST_SEC)
    value['start_decision'] = Q.decide(value, debounce_sec=Q.DEBOUNCE_SEC)
    # 旧方式の資格を新方式へ偽装しない。人工再構成の旧anchorは使わない。
    value['start_anchor'] = None
    return value


def trace(value: dict) -> list[dict]:
    changes = {r['frame_idx']: r for r in value['accounting']['rows']}
    stable = {(r['frame_idx'], r['side']): r['arguments'] for r in value['stable_snapshots']}
    gross = dict(value['accounting']['initial_gross_counters'])
    pending, finalized, result = [0, 0], [0, 0], []
    for frame in range(value['first_frame'], value['last_frame'] + Q.STRIDE, Q.STRIDE):
        if frame in changes:
            change = changes[frame]
            gross = {k: v + change['deltas'][k] for k, v in gross.items()}
            pending = [change['pending_after'][s] for s in S.SIDES]
            for attack in change['attack_finalizations']:
                finalized[S.SIDES.index(attack['side'])] += 1
        args = [stable.get((frame, s), dict(score=6, bstate=dict(value='menu' if frame == RESET else 'tsumo_fall')))
                for s in ('1P', '2P')]
        activity = {k: [gross[k + '_' + s] for s in S.SIDES] for k in Q.ACTIVITY if k != 'finalized'}
        result.append(dict(frame=frame, game=int(frame >= QUALIFIED), active=frame >= 60,
            gates=[True, True], states=[a['bstate']['value'] for a in args], scores=[a['score'] for a in args],
            resets=[gross['boundary_resets_' + s] for s in S.SIDES], activity=activity | {'finalized': list(finalized)},
            pending=list(pending), capped=list(pending), leftover=[0, 0], unsettled=[False, False]))
    return result


def identity(value: dict) -> tuple:
    return value['identity']['source_id'], value['identity']['run_id'], 'start-join-artificial'


def test_new_start_reaches_original_canonical_save(tmp_path: Path) -> None:
    value = artificial()
    prefix, cutoff, eligible = S.build(value, identity(value))
    assert eligible and value['start_decision']['evidence']['scores'] == [6, 6]
    with S.J.K.Connection(tmp_path / 'events.jsonl', identity(value)) as connection:
        saved = connection.accept(prefix, cutoff)
    assert not saved.observation.quality.quarantined
    assert saved.observation.cutoff_frame == value['last_frame']


def test_legacy_original_build_is_unchanged() -> None:
    value = original()
    actual, cutoff, eligible = S.build(value, identity(value))
    expected, previous, was_eligible = S.J.build(value, identity(value))
    assert actual == expected and cutoff == previous and eligible == was_eligible


@pytest.mark.parametrize('case,reason', [('pending', 'trace_pending_mismatch'),
    ('finalized', 'trace_finalized_mismatch'), ('metadata', 'trace_metadata_mismatch'),
    ('decision', 'saved_start_decision'), ('identity', 'source_run')])
def test_consumer_rejects_independent_mismatch(case: str, reason: str) -> None:
    value = artificial()
    scope = identity(value)
    if case in ('pending', 'finalized'):
        last = value['start_counter_trace'][-1]
        target = last['pending'] if case == 'pending' else last['activity']['finalized']
        target[0] += 1
    elif case == 'metadata':
        value['start_counter_trace'][-1]['scores'][0] += 1
    elif case == 'decision':
        value['start_decision']['eligible'] = False
    else:
        scope = ('different-source', scope[1], scope[2])
    with pytest.raises(ValueError, match=reason):
        S.build(value, scope)
