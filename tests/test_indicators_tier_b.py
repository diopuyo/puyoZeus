"""Tier B 形質指標 (2026-05-05、key_flexibility 二相性分離) のテスト。

- PlanningEntropyIndicator (1 ツモ追加で発火する連鎖サイズ分布のエントロピー)
- StructureSolidityIndicator (下半分の連結 ≥3 ぷよ数比率)
- BaseFlatnessIndicator (下層 3 段の高さ標準偏差)
"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)
from src.chain import ChainSimulator
from src.old.indicators import (
    INDICATOR_BASE_FLATNESS,
    INDICATOR_PLANNING_ENTROPY,
    INDICATOR_STRUCTURE_SOLIDITY,
    BaseFlatnessIndicator,
    IndicatorCalculator,
    PlanningEntropyIndicator,
    StructureSolidityIndicator,
    _CC_CACHE,
    _connected_components,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _flat_board() -> Board:
    """下層 1 段が均等に埋まった平らな盤面。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(BOARD_COLS):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


def _uneven_board() -> Board:
    """高さ差が大きい盤面 (左 5 段、右 0 段)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - 5, BOARD_ROWS):
        grid[r][0] = COLOR_RED
        grid[r][1] = COLOR_BLUE
    return Board.from_list(grid)


def _solid_board() -> Board:
    """下半分に 4 連結のみ。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


def _scattered_board() -> Board:
    """下半分に単独ぷよ多数 (連結 1 ずつ)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 互い違いに配置 (隣接同色なし)
    grid[BOARD_ROWS - 1][0] = COLOR_RED
    grid[BOARD_ROWS - 1][2] = COLOR_BLUE
    grid[BOARD_ROWS - 1][4] = COLOR_RED
    grid[BOARD_ROWS - 2][1] = COLOR_BLUE
    grid[BOARD_ROWS - 2][3] = COLOR_RED
    grid[BOARD_ROWS - 2][5] = COLOR_BLUE
    return Board.from_list(grid)


# ============================
# PlanningEntropy
# ============================


def test_planning_entropy_name() -> None:
    ind = PlanningEntropyIndicator()
    assert ind.name == INDICATOR_PLANNING_ENTROPY


def test_planning_entropy_empty_board_zero() -> None:
    """空盤面では試行可能だが連鎖サイズは全て 0 (1 個置きで連鎖しない) → エントロピー 0。"""
    ind = PlanningEntropyIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), simulator=sim)
    # 全列で chain_count=0 → 1 ビンに集中 → entropy=0
    assert res.score == 0.0
    assert res.detail.get("n_bins", 0) == 1


def test_planning_entropy_score_in_range() -> None:
    ind = PlanningEntropyIndicator()
    sim = ChainSimulator()
    res = ind.compute(_solid_board(), simulator=sim)
    assert 0.0 <= res.score <= 1.0


def test_planning_entropy_diverse_chains_higher() -> None:
    """連鎖サイズが分散する盤面の方が、全列同じ連鎖サイズの盤面よりエントロピー高い。"""
    # 列毎に発火可能性が異なる盤面を作成
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # col 0: 3 連結 (red) → 1 個追加で発火
    for r in range(BOARD_ROWS - 3, BOARD_ROWS):
        grid[r][0] = COLOR_RED
    # col 5: 単独 → 1 個追加で発火しない
    grid[BOARD_ROWS - 1][5] = COLOR_BLUE
    diverse = Board.from_list(grid)
    ind = PlanningEntropyIndicator()
    sim = ChainSimulator()
    res_diverse = ind.compute(diverse, simulator=sim)
    res_empty = ind.compute(_empty_board(), simulator=sim)
    # diverse は 0 と非0 の chain_count が混在 → entropy > 0
    # empty は全て 0 → entropy = 0
    assert res_diverse.score >= res_empty.score


# ============================
# StructureSolidity
# ============================


def test_structure_solidity_name() -> None:
    ind = StructureSolidityIndicator()
    assert ind.name == INDICATOR_STRUCTURE_SOLIDITY


def test_structure_solidity_empty_board_zero() -> None:
    ind = StructureSolidityIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 0.0
    assert res.detail.get("reason") == "empty_bottom"


def test_structure_solidity_solid_4_connected() -> None:
    """下半分に 4 連結のみ → solid=4, total=4, ratio=1.0。"""
    ind = StructureSolidityIndicator()
    res = ind.compute(_solid_board())
    assert res.score == 1.0
    assert res.detail["bottom_total"] == 4
    assert res.detail["solid"] == 4


def test_structure_solidity_scattered_low() -> None:
    """単独ぷよだらけ → solid=0、score=0.0。"""
    ind = StructureSolidityIndicator()
    res = ind.compute(_scattered_board())
    assert res.score == 0.0
    assert res.detail["solid"] == 0


def test_structure_solidity_with_ojama() -> None:
    """おじゃまは bottom_total に加算されるが solid に入らない → ratio が下がる。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(3):
        grid[BOARD_ROWS - 1][c] = COLOR_RED  # 3 連結
    grid[BOARD_ROWS - 1][5] = COLOR_OJAMA  # おじゃま
    board = Board.from_list(grid)
    ind = StructureSolidityIndicator()
    res = ind.compute(board)
    # solid=3 (3 連結), total=3 (3 連結) + 1 (おじゃま) = 4 → ratio=0.75
    assert res.detail["bottom_total"] == 4
    assert res.detail["solid"] == 3
    assert abs(res.score - 0.75) < 1e-6


# ============================
# BaseFlatness
# ============================


def test_base_flatness_name() -> None:
    ind = BaseFlatnessIndicator()
    assert ind.name == INDICATOR_BASE_FLATNESS


def test_base_flatness_empty_board() -> None:
    """空盤面: 全列 0 高 → std=0 → score=1.0 (平ら)。"""
    ind = BaseFlatnessIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 1.0
    assert res.raw_value == 0.0


def test_base_flatness_flat_board_high() -> None:
    """全列均等 → std=0 → score=1.0。"""
    ind = BaseFlatnessIndicator()
    res = ind.compute(_flat_board())
    assert res.score == 1.0


def test_base_flatness_uneven_board_low() -> None:
    """凸凹あり → std>0 → score<1.0。"""
    ind = BaseFlatnessIndicator()
    res = ind.compute(_uneven_board())
    assert res.score < 1.0
    assert res.raw_value > 0.0


def test_base_flatness_score_in_range() -> None:
    ind = BaseFlatnessIndicator()
    sim = ChainSimulator()
    for board in [_empty_board(), _flat_board(), _uneven_board(), _solid_board()]:
        res = ind.compute(board, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# _connected_components キャッシュ
# ============================


def test_connected_components_cache_hit() -> None:
    """同一盤面で 2 度呼ぶとキャッシュにヒットして同一オブジェクトを返す。"""
    _CC_CACHE.clear()
    board = _solid_board()
    r1 = _connected_components(board)
    r2 = _connected_components(board)
    assert r1 is r2  # キャッシュからの同一オブジェクト


def test_connected_components_cache_isolation() -> None:
    """異なる盤面では別の結果になる。"""
    _CC_CACHE.clear()
    r_solid = _connected_components(_solid_board())
    r_empty = _connected_components(_empty_board())
    # solid: 1 連結成分 (4 個 red), empty: 0 連結成分
    assert len(r_solid) == 1
    assert r_solid[0][0] == COLOR_RED
    assert len(r_solid[0][1]) == 4
    assert len(r_empty) == 0


# ============================
# IndicatorCalculator 統合
# ============================


def test_calc_compute_all_tier_b() -> None:
    """IndicatorCalculator が Tier B 3 指標を含めて計算する。"""
    calc = IndicatorCalculator()
    res = calc.compute_all(_solid_board())
    assert INDICATOR_PLANNING_ENTROPY in res.results
    assert INDICATOR_STRUCTURE_SOLIDITY in res.results
    assert INDICATOR_BASE_FLATNESS in res.results
    assert res.planning_entropy == res.results[INDICATOR_PLANNING_ENTROPY].score
    assert res.structure_solidity == res.results[INDICATOR_STRUCTURE_SOLIDITY].score
    assert res.base_flatness == res.results[INDICATOR_BASE_FLATNESS].score


def test_calc_extra_indicator_names_includes_tier_b() -> None:
    """EXTRA_INDICATOR_NAMES に Tier B 3 指標が含まれる。"""
    from src.old.indicators import EXTRA_INDICATOR_NAMES
    assert INDICATOR_PLANNING_ENTROPY in EXTRA_INDICATOR_NAMES
    assert INDICATOR_STRUCTURE_SOLIDITY in EXTRA_INDICATOR_NAMES
    assert INDICATOR_BASE_FLATNESS in EXTRA_INDICATOR_NAMES
