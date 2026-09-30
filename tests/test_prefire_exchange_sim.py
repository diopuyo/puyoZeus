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
    """3列目以外は隠し段まで埋まり、3列目に赤3個。赤ペアは3列目に縦置きしかできず、置けば可視4個で消える。"""
    grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_OJAMA, dtype=np.int8)
    grid[0:2, 2] = 0
    grid[2:5, 2] = RED
    options = sim.fire_options(raw(grid), (RED, RED, BLUE, BLUE))
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
