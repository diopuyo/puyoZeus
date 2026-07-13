"""有利不利オーバーレイの合成成分(圧力/得点リード)の単体テスト。

PressureTracker(着弾ダメージの持続記憶)と ScoreLeadTracker(得点リード)は
純ロジックなので認識なしで検証できる。回帰防止用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import (  # noqa: E402
    PressureTracker, ScoreLeadTracker,
)


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
