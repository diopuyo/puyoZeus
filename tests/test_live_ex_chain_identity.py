"""EXの短い再通知で台帳の基準を作り直さない試作を検査する。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j._live_ex_frozen.src import prefire_chain_identity as identity

START = 10.0
BEFORE = 20.0
AFTER = BEFORE + 2 / 30
COUNT = 15
BOARD = {'board': {'array': [[1, 2, 3, 4, 0, 0]]}}


def sample() -> tuple[dict, dict, dict]:
    event = dict(trigger_sec=START, chain_count=COUNT, total_score=112100, before_board=BOARD)
    source = dict(tsumo_count=73, next_slide_motion=False)
    chain = dict(signature=(START, COUNT, 112100), started=START, generated_at_fire=1000, finished=False)
    identity.remember(chain, event, BEFORE, source)
    return chain, dict(event, trigger_sec=AFTER, total_score=113180), source


def test_default_off_does_not_change_input_or_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(identity.ENV, raising=False)
    chain, event, source = sample()
    original = deepcopy(chain)
    assert identity.normalize(chain, event, AFTER, source) is event
    assert chain == original and identity.STATE_KEY not in chain


def test_same_count_short_gap_keeps_origin_and_next_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    original = deepcopy(event)
    assert identity.normalize(chain, event, AFTER, source)['trigger_sec'] == START
    assert identity.normalize(chain, event, AFTER + 1 / 30, source)['trigger_sec'] == START
    assert chain['started'] == START and chain['generated_at_fire'] == 1000
    assert event == original


@pytest.mark.parametrize('change', ['finished', 'lower_count', 'single', 'long_gap',
                                  'placed', 'next', 'external', 'future'])
def test_does_not_merge_uncertain_or_new_chain(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    stamp = AFTER
    if change in ('finished', 'external'):
        chain[change] = True
    elif change in ('lower_count', 'single'):
        event['chain_count'] = {'lower_count': COUNT - 1, 'single': 1}[change]
    elif change == 'long_gap':
        stamp = BEFORE + identity.MAX_GAP_SEC + 1 / 30
    elif change == 'placed':
        source['tsumo_count'] += 1
    elif change == 'next':
        source['next_slide_motion'] = True
    elif change == 'future':
        event['trigger_sec'] = AFTER + 1
    assert identity.normalize(chain, event, stamp, source)['trigger_sec'] == event['trigger_sec']


def test_training_without_tsumo_accepts_same_short_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    source.pop('tsumo_count')
    assert identity.normalize(chain, event, AFTER, source)['trigger_sec'] == START


def test_chain_update_does_not_reinitialize_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j._live_ex_frozen.src import prefire_post_b3 as module
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    chain.update(pool=[dict(log_weight=0)], observed=[])
    subject = SimpleNamespace(chains=[chain, None], fight=START, native=None, policy={},
                              state=SimpleNamespace(colors=SimpleNamespace(known=lambda: (None, (1, 2, 3, 4)))),
                              _initial=lambda *args: pytest.fail('同じ連鎖の台帳を初期化した'))
    monkeypatch.setattr(module, 'advance', lambda *args: {})
    monkeypatch.setattr(module, 'error', lambda *args: 0)
    raw = [dict(source, state=dict(state='CHAIN'), chain_event=dict(namespace=event)),
           dict(state=dict(state='STABLE'))]
    module.PostB3._chains(subject, AFTER, raw)
    assert subject.chains[0] is chain
    assert chain['started'] == START and chain['generated_at_fire'] == 1000
    assert chain['signature'] == (START, COUNT, event['total_score'])


def test_scene4_recorded_packets_keep_original_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    path = Path(__file__).parent / 'fixtures/live_ex_chain_identity_scene4.json'
    before, after = json.loads(path.read_text('utf-8'))['observations']
    first = before['raw']['chain_event']['namespace']
    event = after['raw']['chain_event']['namespace']
    chain = dict(signature=(first['trigger_sec'], first['chain_count'], first['total_score']),
                 started=first['trigger_sec'], generated_at_fire=1000, finished=False)
    identity.remember(chain, first, before['t'], before['raw'])
    corrected = identity.normalize(chain, event, after['t'], after['raw'])
    assert corrected['trigger_sec'] == first['trigger_sec'] != event['trigger_sec']
    assert chain['started'] == first['trigger_sec'] and chain['generated_at_fire'] == 1000


def test_progression_and_noisy_board_within_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    event.update(chain_count=COUNT + 1, before_board=None)
    assert identity.normalize(chain, event, BEFORE + 19 / 30, source)['trigger_sec'] == START


@pytest.mark.parametrize('old_count,new_count', [(1, 2), (2, 3), (2, 5)])
def test_early_progression_is_ambiguous(old_count: int, new_count: int,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(identity.ENV, '1')
    chain, event, source = sample()
    chain[identity.STATE_KEY]['count'] = old_count
    event['chain_count'] = new_count
    assert identity.normalize(chain, event, AFTER, source)['trigger_sec'] == AFTER
