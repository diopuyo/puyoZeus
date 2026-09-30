"""1 手展開の高速化 (出力同一)。連鎖が起きない子盤面では、盤面全体の連結検出を省く。

近未来探索 (`indicators_v2._near_future_*_expand`) と応手探索 (`exchange_event_multilanding`) は、
基準盤面 (直前の連鎖が終わった静止盤面) に 1〜2 個のぷよを足した子盤面を大量に作り、
その都度 `ChainSimulator.simulate` で盤面全体の連結を調べる。子盤面の大半は連鎖しない。

基準盤面に 4 個以上の連結が無い (= 静止) なら、子盤面で新しくできる 4 連結は必ず
「今回足したセルを含む連結」なので、足したセルの周りだけを調べれば連鎖の有無が確定する。
- 連鎖なし: `simulate` の結果は 連鎖数 0・消去 0・得点 0・final_board = 子盤面そのもの。
- 連鎖あり (または基準盤面が静止でない): 従来どおり `simulate` を呼ぶ。
幽霊連鎖ルール (隠し段のセルは連結に入らない) は `ChainSimulator` の設定をそのまま使う。
列挙の順序・空き判定・ソートは `indicators_v2._enumerate_placements` / `_drop_one_color` と同一。
"""
from __future__ import annotations

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, HIDDEN_ROWS, Board
from src.chain import MIN_ERASE_COUNT, NEIGHBOR_DELTAS, ChainResult, ChainSimulator
from src.exchange_multilanding_precheck import column_heights
from src.scoring import calculate_chain_score

Cell = tuple[int, int]
VERTICAL_ROTATIONS = (0, 2)   # 縦置き (rotation 0: TOP 上 / 2: BOT 上)
ROTATIONS = (0, 1, 2, 3)
VERTICAL_HEADROOM = BOARD_ROWS - 2   # 縦置きが可能な最大の列高さ (`_drop_two_in_column` と同じ)
NO_CHAIN_SCORE = 0                   # 連鎖なしの素点 (`calculate_chain_score` の空の連鎖)


def new_board(grid: np.ndarray) -> Board:
    """空盤面を一度作って捨てる `Board()` の無駄を避け、渡された配列をそのまま持つ Board。"""
    board = Board.__new__(Board)
    board._grid = grid
    return board


def local_pop(grid: np.ndarray, cells: list[Cell], hidden: int) -> bool:
    """足した `cells` を含む連結のどれかが MIN_ERASE_COUNT 個以上か (連結の規則は ChainSimulator と同一)。

    hidden: 連結に入れない上端の行数 (幽霊連鎖ルール ON なら HIDDEN_ROWS、OFF なら 0)。
    """
    seen: set[Cell] = set()
    for start in cells:
        if start[0] < hidden or start in seen:
            continue
        color = grid.item(*start)
        seen.add(start)
        stack, size = [start], 0
        while stack:
            row, col = stack.pop()
            size += 1
            if size >= MIN_ERASE_COUNT:
                return True
            for dr, dc in NEIGHBOR_DELTAS:
                nr, nc = row + dr, col + dc
                if (hidden <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS and (nr, nc) not in seen
                        and grid.item(nr, nc) == color):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return False


def pair_cells(heights: list[int], pair: tuple[int, int], col: int, rotation: int) -> list[tuple[int, int, int]] | None:
    """`_place_pair_to_board` と同じ規則で、足すセル (row, col, color) を返す。置けなければ None。"""
    top, bot = pair
    if top == COLOR_EMPTY or bot == COLOR_EMPTY:
        return None
    if rotation in VERTICAL_ROTATIONS:
        if heights[col] > VERTICAL_HEADROOM:
            return None
        upper, lower = (top, bot) if rotation == 0 else (bot, top)
        floor = BOARD_ROWS - 1 - heights[col]
        return [(floor, col, lower), (floor - 1, col, upper)]
    if heights[col] >= BOARD_ROWS or heights[col + 1] >= BOARD_ROWS:
        return None
    left, right = (top, bot) if rotation == 1 else (bot, top)
    return [(BOARD_ROWS - 1 - heights[col], col, left), (BOARD_ROWS - 1 - heights[col + 1], col + 1, right)]


class FastExpander:
    """`ChainSimulator` 1 個に紐づく、出力同一の高速展開器。"""

    def __init__(self, simulator: ChainSimulator) -> None:
        self.sim = simulator
        self.hidden = HIDDEN_ROWS if simulator._exclude_hidden_row_from_pop else 0

    def is_stable(self, board: Board) -> bool:
        """連鎖が起きない静止盤面か (局所判定の前提)。"""
        return not self.sim.find_erasable_groups(board)

    def child(self, base: Board, stable: bool, cells: list[tuple[int, int, int]]) -> tuple[Board, ChainResult | None]:
        """足したセルを持つ子盤面と、連鎖があった場合だけその結果 (なければ None)。"""
        grid = base._grid.copy()
        for row, col, color in cells:
            grid[row, col] = color
        placed = new_board(grid)
        if stable and not local_pop(grid, [(r, c) for r, c, _ in cells], self.hidden):
            return placed, None
        return placed, self.sim.simulate(placed)

    def placements(self, base: Board, pair: tuple[int, int],
                   stable: bool | None = None) -> list[tuple[int, Board, ChainResult | None]]:
        """`_enumerate_placements` と同じ順序の (連鎖数, 配置後盤面, 連鎖結果 or None)。

        stable: 基準盤面の静止判定を呼出し側で持っている場合に渡す (同じ盤面で複数ペアを試すとき)。
        """
        heights = [int(v) for v in column_heights(base._grid)]
        stable = self.is_stable(base) if stable is None else stable
        rows: list[tuple[int, Board, ChainResult | None]] = []
        for rotation in ROTATIONS:
            for col in range(BOARD_COLS if rotation in VERTICAL_ROTATIONS else BOARD_COLS - 1):
                cells = pair_cells(heights, pair, col, rotation)
                if cells is None:
                    continue
                placed, result = self.child(base, stable, cells)
                rows.append((0 if result is None else result.chain_count, placed, result))
        rows.sort(key=lambda x: x[0], reverse=True)
        return rows

    def score_of(self, result: ChainResult | None, use_exact_score: bool) -> float:
        """得点。連鎖なしは 0、連鎖ありは従来と同じ式。"""
        if result is None:
            return float(NO_CHAIN_SCORE)
        if use_exact_score and hasattr(result, 'exact_score'):
            return float(result.exact_score)
        return float(calculate_chain_score(result).total_score)

    def known(self, frontier: list, pair: tuple[int, int], resolve_before_death: bool,
              use_exact_score: bool) -> list[tuple[float, Board, int]]:
        """`_near_future_known_expand` の候補列 (ソート前)。"""
        candidates: list[tuple[float, Board, int]] = []
        for _, base in frontier:
            for count, placed, result in self.placements(base, pair):
                if not resolve_before_death and placed.is_dead():
                    continue
                final = placed if result is None else result.final_board
                if resolve_before_death and final.is_dead():
                    continue
                candidates.append((self.score_of(result, use_exact_score), final, count))
        return candidates

    def free(self, frontier: list, colors: tuple[int, ...], resolve_before_death: bool,
             use_exact_score: bool) -> list[tuple[float, Board, int]]:
        """`_near_future_free_expand` の候補列 (ソート前)。"""
        candidates: list[tuple[float, Board, int]] = []
        for _, base in frontier:
            heights = [int(v) for v in column_heights(base._grid)]
            stable = self.is_stable(base)
            for col in range(BOARD_COLS):
                if heights[col] >= BOARD_ROWS:
                    continue
                for color in colors:
                    placed, result = self.child(base, stable, [(BOARD_ROWS - 1 - heights[col], col, color)])
                    if not resolve_before_death and placed.is_dead():
                        continue
                    final = placed if result is None else result.final_board
                    if resolve_before_death and final.is_dead():
                        continue
                    candidates.append((self.score_of(result, use_exact_score), final,
                                       0 if result is None else result.chain_count))
        return candidates


def drop_ojama_fast(board: Board, ojama_count: int, remainder_columns: tuple[int, ...]) -> Board:
    """`ChainSimulator.drop_ojama_with_remainder_columns` と同じ結果 (セル参照を配列操作へ置換)。

    各列で、下から順に空きセルへ `落下数` 個のおじゃまを置く (置けない分は捨てる = 窒息判定側で検出)。
    列数・重複・範囲の検証は従来と同じ関数で行う。
    """
    counts = ChainSimulator._calc_explicit_ojama_drop_counts(ojama_count, tuple(remainder_columns))
    grid = board._grid.copy()
    for col, remaining in enumerate(counts):
        if remaining <= 0:
            continue
        empty_from_bottom = np.flatnonzero(grid[:, col] == COLOR_EMPTY)[::-1]
        grid[empty_from_bottom[:remaining], col] = COLOR_OJAMA
    return new_board(grid)
