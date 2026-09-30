"""発火前の撃ち合いシミュレータ (src/prefire_exchange_sim.py) の単体テスト (段1)。"""
from __future__ import annotations

import numpy as np
import pytest

from src import prefire_exchange_sim as sim
from src import puyo_core_bridge as native
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA

RED, BLUE, GREEN, YELLOW = 1, 2, 3, 4

pytestmark = pytest.mark.skipif(not native.NATIVE_AVAILABLE, reason='puyo_core 未ビルド')


def raw(grid: np.ndarray) -> bytes:
    return grid.astype(np.int8).tobytes()


def empty() -> np.ndarray:
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)


def three_reds() -> np.ndarray:
    """左端に赤3個 (赤ペアを縦に置けば 5 個で1連鎖)。"""
    grid = empty()
    grid[10:13, 0] = RED
    return grid


def test_invalid_queue_is_missing_not_searched() -> None:
    options = sim.fire_options(raw(three_reds()), (9, 9, 9, 9))
    assert options == sim.NO_OPTIONS and not options.queue_known


def test_first_hand_fire_found_and_not_forced() -> None:
    options = sim.fire_options(raw(three_reds()), (RED, RED, BLUE, GREEN))
    assert options.best1 is not None and options.best1.chain_count == 1 and options.best1.score > 0
    assert not options.forced      # 空きが多いので撃たない配置でも生き残る


def test_second_hand_fire_uses_dnext() -> None:
    options = sim.fire_options(raw(three_reds()), (BLUE, GREEN, RED, RED))
    assert options.best1 is None or options.best1.score < options.best2.score
    assert options.best2 is not None and options.best2.hand == 2


def test_forced_when_every_quiet_placement_dies() -> None:
    """3列目以外は隠し段まで埋まり、3列目に赤3個。赤青ペアは3列目に縦置きしかできない。
    赤を下にすれば可視4個で消えて生き残り、青を下にすれば消えずに窒息する (= 撃たないと窒息)。"""
    grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_OJAMA, dtype=np.int8)
    grid[0:2, 2] = 0
    grid[2:5, 2] = RED
    options = sim.fire_options(raw(grid), (RED, BLUE, BLUE, BLUE))
    assert options.best1 is not None
    assert options.forced


def test_hazard_features_bounded_and_ordered() -> None:
    own = sim.fire_options(raw(three_reds()), (RED, RED, BLUE, GREEN))
    values = sim.hazard_features(own, sim.NO_OPTIONS, three_reds(), empty(), 120.0)
    assert values.shape == (len(sim.HAZARD_COLUMNS),)
    assert np.all((values >= 0) & (values <= 1))
    assert values[sim.HAZARD_COLUMNS.index('own_can1')] == 1.0
    assert values[sim.HAZARD_COLUMNS.index('opp_can1')] == 0.0


def test_fire_options_is_pure_and_cached() -> None:
    grid = three_reds()
    first = sim.fire_options(raw(grid), (RED, RED, BLUE, GREEN))
    assert np.array_equal(grid, three_reds())          # 入力を破壊しない
    assert sim.fire_options(raw(grid), (RED, RED, BLUE, GREEN)) is first


def test_board_that_already_pops_is_not_searched() -> None:
    """置く前から4個つながっている盤面 (誤読・連鎖途中) は、どの配置も「発火」に見えるので探索しない。"""
    grid = empty()
    grid[9:13, 0] = RED
    options = sim.fire_options(raw(grid), (BLUE, GREEN, BLUE, GREEN))
    assert options == sim.UNSTABLE_BOARD and options.best() is None and not options.forced


def test_already_dead_board_is_not_forced() -> None:
    """窒息セルが既に埋まった盤面では、どの非発火配置も「窒息」に見える。撃たないと窒息とは言えないので探索しない。"""
    grid = three_reds()
    grid[1:13, 2] = [BLUE if r % 2 else GREEN for r in range(1, 13)]   # 縦に交互 (消える群は作らない)
    options = sim.fire_options(raw(grid), (RED, RED, BLUE, GREEN))
    assert options == sim.UNSTABLE_BOARD and not options.forced


def test_every_placement_pops_is_not_forced() -> None:
    """どこに置いても消える (選択ではない) 盤面は「撃たないと窒息」ではない。"""
    grid = empty()
    grid[12, 1:4] = RED
    grid[10:13, 5] = RED
    grid[10:12, 0] = RED
    grid[11:13, 4] = BLUE
    options = sim.fire_options(raw(grid), (RED, RED, BLUE, GREEN))
    assert options.best1 is not None and not options.forced


def test_counter_scores_reproducible_and_missing_is_zero() -> None:
    grid = three_reds()
    colors = (RED, BLUE, GREEN, YELLOW)
    first = sim.counter_scores(raw(grid), (RED, RED, BLUE, GREEN), 3, colors, seed=7)
    again = sim.counter_scores(raw(grid), (RED, RED, BLUE, GREEN), 3, colors, seed=7)
    assert first.shape == (sim.COUNTER_ROLLOUTS,) and np.array_equal(first, again)
    assert (first > 0).all()     # 既知 NEXT の赤ペアで必ず消せる = 最善応手は全標本で撃つ
    assert not sim.counter_scores(raw(grid), (9, 9, 9, 9), 3, colors, seed=7).any()


def test_seen_colors_only_playable() -> None:
    grid = three_reds()
    grid[12, 5] = COLOR_OJAMA
    assert sim.seen_colors([grid], [(BLUE, GREEN, 9, 9)]) == (RED, BLUE, GREEN)
