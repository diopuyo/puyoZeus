"""発火前オラクル測定器 (scripts/prefire_oracle_ceiling_20260930.py) の注入規則の単体テスト (段1)。"""
from __future__ import annotations

import numpy as np

from scripts import prefire_oracle_ceiling_20260930 as oracle

FPS = 30


def make_display(seconds: float = 10.0) -> dict:
    """1試合・一定勝率 0.4 の表示列。"""
    t = np.arange(0, seconds, 1 / FPS)
    return dict(t_sec=t, game_idx=np.zeros(len(t), dtype=int), display_p1=np.full(len(t), 0.4),
                display_adv=np.full(len(t), -20.0))


def row(trigger: float, prev_close: float, target: float) -> dict:
    return dict(exchange_id=1, game=0, trigger=trigger, close=trigger + 2, prev_close=prev_close,
                target=target, at_fire=0.4)


def test_blend_zero_is_identity() -> None:
    display = make_display()
    p1, mask = oracle.inject(display, [row(5.0, 0.0, 0.9)], 3.0, 0.0)
    np.testing.assert_allclose(p1, display['display_p1'])
    assert mask.sum() == 3 * FPS


def test_full_blend_sets_target_only_before_fire() -> None:
    display = make_display()
    p1, mask = oracle.inject(display, [row(5.0, 0.0, 0.9)], 1.0, 1.0)
    times = display['t_sec']
    assert np.allclose(p1[mask], 0.9)
    assert times[mask].min() >= 4.0 - 1e-9 and times[mask].max() < 5.0
    assert np.allclose(p1[~mask], 0.4)   # 発火以後は書き換えない (撃ち合い中は現行の反応のまま)


def test_window_is_clipped_at_previous_exchange_close() -> None:
    display = make_display()
    _, mask = oracle.inject(display, [row(5.0, 4.5, 0.9)], 3.0, 1.0)
    assert display['t_sec'][mask].min() >= 4.5 - 1e-9


def test_placebo_permutes_targets_without_changing_multiset() -> None:
    rows = [row(float(i), 0.0, t) for i, t in enumerate((0.1, 0.2, 0.7, 0.9))]
    shuffled = oracle.placebo_rows(rows)
    assert sorted(r['target'] for r in shuffled) == sorted(r['target'] for r in rows)
    assert [r['trigger'] for r in shuffled] == [r['trigger'] for r in rows]


def test_freshness_anticipation_ratio() -> None:
    display = make_display()
    display['display_p1'][(display['t_sec'] >= 3.0)] = 0.6   # 発火2秒前に目標方向へ半分動く
    item = oracle.window_freshness(display, row(5.0, 0.0, 0.8), 3.0)
    assert item is not None
    assert abs(item['anticipation'] - 0.5) < 1e-9
    assert abs(item['jump_after'] - 0.2) < 1e-9
