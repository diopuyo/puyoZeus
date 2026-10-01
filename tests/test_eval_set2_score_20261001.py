"""セット2の確定監査は内部試合番号でなく公式時刻窓で照合する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

from scripts import eval_set2_score_20261001 as target


def test_death_audit_maps_time_and_counts_each_side_once(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """別区間で同じ内部番号でも公式試合を混同せず、繰返し通知を二重計上しない。"""
    monkeypatch.setattr(target, 'OUT', tmp_path)
    dest = tmp_path/'replay/e19/s1'
    dest.mkdir(parents=True)
    values = [dict(source='unavoidable_death', t_sec=t, dead_sides=['1P']) for t in (9.,10.,11.,20.)]
    (dest/'events.jsonl').write_text(json.dumps(dict(game_idx=0, values=values)))
    games = [dict(game=1, start=10.,end=20.,winner='2P'),
             dict(game=2, start=20.,end=30.,winner='1P')]
    result = target.death_audit(['s1'],'e19',games)
    assert result['cases'] == 2 and result['false'] == 1
    assert result['outside_window_values'] == 1
    assert result['rows'][0]['first_sec'] == 10.
    assert result['rows'][1]['game'] == 2


def test_confirmed_display_wrong_side_is_detected(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """確定由来の誤表示と、低確率にすぎない通常表示を区別する。"""
    monkeypatch.setattr(target, 'OUT', tmp_path)
    dest = tmp_path/'replay/e19/s1'
    dest.mkdir(parents=True)
    np.savez(dest/'display.npz', t_sec=[10.,11.,12.], display_p1=[.02,.02,.98],
             source=['confirmed_death','G_fe','unavoidable_death'])
    games = [dict(game=1,start=10.,end=20.,winner='1P')]
    result = target.display_deaths(['s1'],'e19',games)
    assert result['false_frames'] == 1 and result['rows'][0]['first_sec'] == 10.
