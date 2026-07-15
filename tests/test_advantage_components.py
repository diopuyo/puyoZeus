"""有利不利オーバーレイの合成成分(圧力/得点リード)の単体テスト。

PressureTracker(着弾ダメージの持続記憶)と ScoreLeadTracker(得点リード)は
純ロジックなので認識なしで検証できる。回帰防止用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types

import numpy as np

from scripts.visualize_advantage_overlay import (  # noqa: E402
    PressureTracker, ScoreLeadTracker, RealtimeForecastTracker, adv_to_winprob,
    kill_override, board_room,
)


def test_kill_no_effect_when_survivable() -> None:
    """降る量が受け容量に対し小さければ有利不利は不変(通常の攻めは上書きしない)。"""
    assert kill_override(20.0, inc1=10.0, inc2=0.0, room1=60, room2=60) == 20.0


def test_kill_pushes_to_survivor_when_1p_lethal() -> None:
    """1Pに致死量(pending≫空き)が降る → 有利不利は2P側(負)へ強制。"""
    v = kill_override(30.0, inc1=200.0, inc2=0.0, room1=8, room2=60)
    assert v <= -90.0  # ほぼ完全上書き(2P勝ち)


def test_kill_pushes_to_survivor_when_2p_lethal() -> None:
    """2Pに致死量 → 1P側(正)へ。"""
    v = kill_override(-30.0, inc1=0.0, inc2=200.0, room1=60, room2=8)
    assert v >= 90.0


def test_kill_ignores_small_attack() -> None:
    """1ターンで捌ける小さな攻め(pending<40)は空きを超えても致死扱いしない。"""
    assert kill_override(15.0, inc1=25.0, inc2=0.0, room1=12, room2=40) == 15.0


def test_kill_symmetric_mutual_no_override() -> None:
    """双方が同程度に致死(相打ち)なら致死度差が小さく上書きしない。"""
    assert kill_override(5.0, inc1=200.0, inc2=200.0, room1=8, room2=8) == 5.0


def test_board_room_full_and_empty() -> None:
    """空盤面は容量72、埋まった盤面は容量0(row0=隠し段は除外)。"""
    empty = types.SimpleNamespace(_grid=np.zeros((13, 6), dtype=np.uint8))
    full = types.SimpleNamespace(_grid=np.ones((13, 6), dtype=np.uint8))
    assert board_room(empty) == 72
    assert board_room(full) == 0
    assert board_room(None) == 72


def test_winprob_even_is_half() -> None:
    """有利不利0 → 勝率50%(較正sigmoid・直線どちらも対称で成立)。"""
    assert abs(adv_to_winprob(0.0) - 0.5) < 1e-6


def test_winprob_monotonic() -> None:
    """有利不利が上がるほど1P勝率も上がる(単調増加)。"""
    assert adv_to_winprob(-80) < adv_to_winprob(-20) < adv_to_winprob(20) < adv_to_winprob(80)


def test_winprob_range() -> None:
    """勝率は [0,1] に収まる。"""
    for a in (-100.0, -50.0, 0.0, 50.0, 100.0):
        assert 0.0 <= adv_to_winprob(a) <= 1.0


def test_forecast_opponent_attack_negative() -> None:
    """2P が発火(score増)→ 1Pへの pending 増 → 信号は負(2P有利)。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    assert ft.update(0, 8000, 0, 0) < 0.0


def test_forecast_self_attack_positive() -> None:
    """1P が発火(score増)→ 2Pへの pending 増 → 信号は正(1P有利)。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    assert ft.update(8000, 0, 0, 0) > 0.0


def test_forecast_game_reset_clears() -> None:
    """スコア大幅減(試合境界)で pending がクリアされ信号がほぼ0に戻る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    ft.update(0, 8000, 0, 0)  # 2P攻撃で偏り
    v = ft.update(0, 0, 0, 0)  # score 8000→0 のリセット
    assert abs(v) < 1.0


def test_forecast_clamped_range() -> None:
    """予告信号は [-100, 100] に収まる。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    v = ft.update(0, 999999, 0, 0)
    assert -100.0 <= v <= 100.0


def test_forecast_delivery_drains_pending() -> None:
    """1Pが発火→2Pへ pending。2Pがツモを重ねる(配送)と pending が減り信号が0へ寄る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    v0 = ft.update(700, 0, 0, 0)  # 1P発火 → 2Pに pending ~10個 → 正
    assert v0 > 0.0
    v1 = v0
    for k in range(1, 6):  # 2Pがツモを置く=30個/ターン配送で pending が捌ける
        v1 = ft.update(700, 0, 0, k)
    assert abs(v1) < abs(v0)  # 配送で pending 減 → 信号が0へ


def test_forecast_offset_by_counter() -> None:
    """2P発火で1Pに pending。1Pが同等以上を発火(相殺)すると 1P pending が減る。"""
    ft = RealtimeForecastTracker()
    ft.update(0, 0, 0, 0)
    neg = ft.update(0, 7000, 0, 0)  # 2P発火 → 1Pに pending ~100 → 強い負
    after = ft.update(7000, 7000, 0, 0)  # 1Pも同等発火で相殺
    assert after > neg  # 相殺で 1P pending 減 → 信号が負から回復


def test_pressure_zero_when_no_ojama() -> None:
    """お邪魔が両者0のままなら圧力は0。"""
    pt = PressureTracker()
    for _ in range(10):
        assert pt.update(0.0, 0.0) == 0.0


def test_pressure_positive_when_opponent_buried() -> None:
    """相手(2P)盤面のお邪魔が増える=1Pが攻撃を通した→圧力は正(1P有利)。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    val = pt.update(0.0, 20.0)  # 2Pに20個着弾
    assert val > 0.0


def test_pressure_negative_when_self_buried() -> None:
    """自分(1P)盤面のお邪魔が増える→圧力は負(2P有利)。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    val = pt.update(20.0, 0.0)
    assert val < 0.0


def test_pressure_decays_toward_zero() -> None:
    """着弾後、追加が無ければ圧力は減衰して0へ近づく。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    peak = pt.update(0.0, 30.0)
    later = peak
    for _ in range(200):
        later = pt.update(0.0, 30.0)  # 増分なし(同じ値)→減衰のみ
    assert abs(later) < abs(peak)


def test_pressure_clamped_range() -> None:
    """圧力は [-100, 100] に収まる。"""
    pt = PressureTracker()
    pt.update(0.0, 0.0)
    v = pt.update(0.0, 72.0)
    assert -100.0 <= v <= 100.0


def test_score_lead_sign() -> None:
    """スコアが高い側が有利(正=1P)。"""
    lt = ScoreLeadTracker()
    assert lt.update(10000, 2000) > 0.0
    lt2 = ScoreLeadTracker()
    assert lt2.update(2000, 10000) < 0.0


def test_score_lead_even() -> None:
    """同点なら0。"""
    lt = ScoreLeadTracker()
    assert lt.update(5000, 5000) == 0.0


def test_score_lead_none_keeps_last() -> None:
    """score None は直前値を保持する。"""
    lt = ScoreLeadTracker()
    lt.update(10000, 2000)
    v = lt.update(None, None)  # 保持 → 同じ符号
    assert v > 0.0


def test_score_lead_clamped_range() -> None:
    """得点リードは [-100, 100] に収まる。"""
    lt = ScoreLeadTracker()
    v = lt.update(999999, 0)
    assert -100.0 <= v <= 100.0
