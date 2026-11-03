"""表示の選択と公開記録の同値性を確認する。"""
from __future__ import annotations

import pytest

from src.phase_j.live_winrate_switch import select_line, with_switch
from src.phase_j.live_publish import initial_snapshot
from tests.phase_j.test_live_publish import ASSETS
from src.phase_j.validator import validate_snapshot


@pytest.mark.parametrize('v49f,ama,active,expected,source', [
    (None, None, False, .4, 'B3'),
    (.7, None, True, .7, 'v4.9f'),
    (None, .8, False, .4, 'B3'),
    (.7, .8, True, .7, 'v4.9f'),
    (None, .8, True, .4, 'B3'),
    (.7, .8, False, .4, 'B3'),
])
def test_selection(v49f: float | None, ama: float | None, active: bool,
                   expected: float, source: str) -> None:
    values = dict(practical=dict(p1_win_probability=.4),
                  v49f=dict(p1=v49f), ama_plain=dict(p1=ama))
    result = select_line(values, active)
    assert result['line1'] == expected
    assert result['source1'] == source
    assert result['b3'] == .4 and result['ama'] == ama
    assert values['practical']['p1_win_probability'] == .4


def test_off_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIVE_WINRATE_SWITCH', raising=False)
    snapshot = initial_snapshot(ASSETS)
    assert with_switch(snapshot, {'_v49f_active': True}) is snapshot


def test_snapshot_keeps_computations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_WINRATE_SWITCH', '1')
    snapshot = initial_snapshot(ASSETS)
    before = snapshot.to_mapping()
    switched = with_switch(snapshot, {})
    assert validate_snapshot(switched).is_valid
    after = switched.to_mapping()
    assert after['evaluations'].pop('winrate_switch')['source1'] == 'B3'
    assert before == after


def test_confirmed_death_keeps_terminal() -> None:
    values = dict(practical=dict(p1_win_probability=1., source='confirmed_death'),
                  v49f=dict(p1=.7))
    assert select_line(values, True)['line1'] == 1.


def test_record_contains_exact_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_b3_time_gate as gate
    monkeypatch.setenv('PUYO_LIVE_WINRATE_SWITCH', '1')
    records = []
    from types import SimpleNamespace
    monkeypatch.setattr(gate, 'ACTIVE', SimpleNamespace(write=records.append))
    snapshot = with_switch(initial_snapshot(ASSETS), {})
    gate.published(dict(frame=1, t_sec=1.), snapshot, 2.)
    assert records[0]['winrate_switch'] == snapshot.evaluations['winrate_switch']
