"""scripts/b3_ensemble_replay_compare.py (保存済みの B3 入力の行で large と ref 3 本を比べる解析台) の部品の試験。段 1。

台の数字が信用できる前提を、手で作った小さな入力で固定する: 勝者と逆の向きの数え方・時刻合わせ・終了の組の結び付け・
表示の再生 (同じ認識時刻の再評価は前の状態へ戻す・保留で 0.5 へ戻す・死亡の由来は置き換えない)。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import b3_ensemble_replay_compare as rc

FAST_ALPHA = 0.5       # src.exchange_display_smoothing.FAST_EMA_ALPHA


def evaluation(frame: int, t: float, raw: float | None, source: str = 'exchange', game: int = 0, key: tuple = (1, 1)) -> dict:
    return dict(frame=frame, t_sec=t, raw_probability=raw, source=source, game=game, probability=0.5 if raw is None else raw,
                judgment_id=(*key, source, raw, True, game), match_active=True, hold=raw is None)


# ------------------------------------------------------------------------------------ 向きの数え方

def test_wrong_direction_is_opposite_to_the_winner_and_neutral_at_half() -> None:
    p = np.array([0.2, 0.5, 0.8])
    assert rc.wrong_direction(p, '1P').tolist() == [True, False, False]       # 1P 勝者: p < 0.5 が逆
    assert rc.wrong_direction(p, '2P').tolist() == [False, False, True]       # 2P 勝者: p > 0.5 が逆


def test_wrong_seconds_counts_frames_and_keeps_missing_apart() -> None:
    grids = [(np.array([0.2, 0.2, 0.9, np.nan]), '1P', 1), (np.array([0.9, 0.9, 0.1]), '2P', 2)]
    result = rc.wrong_seconds(grids)
    assert result['matches'] == 2 and result['valid_sec'] == pytest.approx(6/rc.FPS)
    assert result['wrong_sec'] == pytest.approx(4/rc.FPS) and result['missing_sec'] == pytest.approx(1/rc.FPS)
    assert result['per_match'][0] == dict(id=1, winner='1P', valid_sec=pytest.approx(3/rc.FPS), wrong_sec=pytest.approx(2/rc.FPS),
                                          missing_sec=pytest.approx(1/rc.FPS))
    assert result['wrong_share'] == pytest.approx(4/6) and rc.wrong_seconds([])['wrong_share'] is None


# ------------------------------------------------------------------------------------ 時刻合わせ・差の統計

def test_value_at_is_the_last_publication_at_or_before_tau() -> None:
    times, values = np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])
    out = rc.value_at(times, values, np.array([0.5, 1.0, 2.5, 9.0]))
    assert np.isnan(out[0]) and out[1:].tolist() == [0.1, 0.2, 0.3]


def test_diff_stats_needs_min_pairs_and_ignores_nan_and_inactive() -> None:
    size = rc.MIN_PAIRS+3                          # NaN 2 つと対象外 1 つを除いて、ちょうど MIN_PAIRS 対が残る
    ref, live = np.full(size, 0.50), np.full(size, 0.60)
    ref[0], live[1] = np.nan, np.nan
    mask = np.ones(size, dtype=bool)
    mask[2] = False
    stats = rc.diff_stats(ref, live, mask)
    assert stats['n'] == rc.MIN_PAIRS and stats['mean'] == pytest.approx(10.0) and stats['over_5pt'] == 1.0 and stats['over_10pt'] == 0.0
    assert rc.diff_stats(ref, live, mask & (np.arange(size) > 3)) == dict(n=size-4)       # MIN_PAIRS 未満は件数だけ返す


# ------------------------------------------------------------------------------------ 終了の組の結び付け

def test_match_table_ties_events_with_the_misc_rule_including_the_late_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """[開始, 終了+8秒]。match_ids が空の検出 (終了の 3.9 秒後) も結ぶ。複数の候補なら開始が最後の試合。8 秒を超えたら結ばない。"""
    matches = [dict(id=1, start=0.0, end=60.0), dict(id=2, start=62.0, end=100.0), dict(id=3, start=200.0, end=260.0)]
    origin = 1000.0
    events = [dict(first=origin+63.9, winner='2P', match_ids=[]), dict(first=origin+99.0, winner='1P', match_ids=[2]),
              dict(first=origin+300.0, winner='1P', match_ids=[])]
    (tmp_path/'tables_C2').mkdir()
    (tmp_path/'tables_C2/matches.json.gz').write_bytes(gzip.compress(json.dumps(matches).encode()))
    (tmp_path/'end_events_C2.json').write_text(json.dumps(events), encoding='utf-8')
    monkeypatch.setattr(rc, 'SCORE_DIR', tmp_path)
    _, winners = rc.match_table(origin)
    assert winners == {2: '1P'}                      # 63.9 は m1 の終了+8 (68) 内だが開始が最後の m2 (62) へ。m3 の後ろ 40 秒の検出は結ばない
    events[0]['first'] = origin+63.0                  # 63.0: 候補は m1 (終了+8=68 以内) と m2 → 開始が最後の m2
    events[1]['first'] = origin+66.0
    (tmp_path/'end_events_C2.json').write_text(json.dumps(events), encoding='utf-8')
    assert rc.match_table(origin)[1] == {2: '1P'}    # 同じ試合に 2 件あれば後の検出が残る (台の規則)
    events[0]['first'] = origin+61.0                  # 61.0: m1 の終了+8 内、m2 はまだ始まらない → m1
    (tmp_path/'end_events_C2.json').write_text(json.dumps(events), encoding='utf-8')
    assert rc.match_table(origin)[1] == {1: '2P', 2: '1P'}


# ------------------------------------------------------------------------------------ 生の確率の置き換えと表示の再生

def test_raw_series_replaces_model_sources_only() -> None:
    run = SimpleNamespace(evals=[evaluation(1, 0.0, 0.3, 'static', key=(1, 10)), evaluation(2, 1/30, 1.0, 'confirmed_death', key=(1, 11)),
                                 evaluation(3, 2/30, None, 'null', key=(1, 12)), evaluation(4, 3/30, 0.4, 'exchange', key=(1, 13))])
    new = {(1, 10): 0.35, (1, 11): 0.99, (1, 13): 0.45}
    assert rc.raw_series(run, None) == [0.3, 1.0, None, 0.4]
    assert rc.raw_series(run, new) == [0.35, 1.0, None, 0.45]          # 死亡の由来と保留は置き換えない


def test_replay_ema_same_tick_reevaluation_restores_the_state_and_hold_resets() -> None:
    rows = [evaluation(1, 0.0, 0.2), evaluation(2, 0.1, 0.6), evaluation(2, 0.1, 0.8), evaluation(3, 0.2, None), evaluation(4, 0.3, 0.9)]
    run = SimpleNamespace(evals=rows)
    shown = rc.replay_shown(run, [r['raw_probability'] for r in rows])
    first = 0.2                                                           # 最初の値は平滑せずそのまま (snap)
    second = FAST_ALPHA*0.8+(1-FAST_ALPHA)*first                          # 同じ時刻 0.1 の再評価 (0.8) が 0.6 の結果を置き換える
    assert shown[0] == pytest.approx(first) and shown[1] == pytest.approx(FAST_ALPHA*0.6+(1-FAST_ALPHA)*first)
    assert shown[2] == pytest.approx(second)
    assert shown[3] == 0.5 and shown[4] == pytest.approx(0.9)             # 保留で平滑を作り直す → 次の値は snap
    new_game = [evaluation(1, 0.0, 0.2), evaluation(2, 0.1, 0.6, game=1)]
    assert rc.replay_shown(SimpleNamespace(evals=new_game), [0.2, 0.6]).tolist() == pytest.approx([0.2, 0.6])   # 試合が変われば snap


def test_replay_agreement_counts_exact_rows() -> None:
    rows = [evaluation(1, 0.0, 0.2), evaluation(2, 0.1, 0.6)]
    result = rc.replay_agreement(SimpleNamespace(evals=rows), np.array([0.2, 0.61]))
    assert result['n'] == 2 and result['exact'] == 1 and result['over_0p001'] == 1 and result['max_pt'] == pytest.approx(1.0)


def test_live_series_keeps_logged_values_when_a_publication_differs_from_its_frame() -> None:
    run = SimpleNamespace(evals=[evaluation(1, 0.0, 0.2), evaluation(2, 0.1, 0.4)],
                          pubs=[dict(frame=1, published_at=10.0, probability=0.2), dict(frame=2, published_at=11.0, probability=0.4),
                                dict(frame=2, published_at=12.0, probability=0.33), dict(frame=9, published_at=13.0, probability=None)])
    out = rc.live_series(run, [0.25, 0.45], dict(origin_mono=5.0), 1.0)
    assert out['times'].tolist() == [4.0, 5.0, 6.0, 7.0]                  # 公開の時刻 − origin_mono − offset
    assert out['values'][:3].tolist() == [0.25, 0.45, 0.33] and np.isnan(out['values'][3])
    assert out['kept_as_logged'] == 2 and out['n'] == 4                    # 保存の値のまま残した公開 (評価行と食い違う・値なし)


def test_bootstrap_mean_is_deterministic_and_small_samples_report_only_n() -> None:
    assert rc.bootstrap_mean([1.0, 2.0]) == dict(n=2)
    a, b = rc.bootstrap_mean([1.0, 2.0, 3.0, 4.0]), rc.bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert a == b and a['lo'] <= a['mean'] <= a['hi'] and a['mean'] == 2.5
