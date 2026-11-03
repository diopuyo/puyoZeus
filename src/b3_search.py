"""B3専用の2個組理想ツモ探索。Qは手数内に発火した最大得点。"""
from __future__ import annotations
from src.b3_profile import timed

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src import indicators_v2 as iv
from src import puyo_core_bridge as native
from src.board import Board, BOARD_COLS, BOARD_ROWS
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import compute_effective_rate

# 2026-10-06 user決定。深さ調査: share/tasks/done/l_b3_depth_probe。
HMAX_TOTAL_HANDS = 20
BOARD_CAPACITY = BOARD_ROWS * BOARD_COLS
# 2026-10-06採用: 固定500件の不足解消73.2%、公開500状態の特徴時間門を通過。
# 全候補と幅8/16/32も計測。根拠: b3_data/codex_tasks/search_redesign.CANDIDATES.csv。
BEAM_WIDTH = 4
POTENTIAL_SHORTLIST = 12
SEARCH_VERSION = "b3_pair_potential_v2"
MATCH_COLOR_COUNT = 4
SEARCH_CACHE_SIZE = 16384
ALL_COLORS: tuple[int, ...] = iv.IGNITION_TRIAL_COLORS


class _NativeSim:
    """学習・本番と同じ native 連鎖計算 (幽霊連鎖規則込み)。"""

    def simulate(self, board: Board) -> native.ChainSimResult:
        return native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


_SIM = _NativeSim()


@dataclass(frozen=True)
class IdealSearchResult:
    """h=1..hmax の最良生得点と、そのときの連鎖数。done_hands=ビームが生存した手数。"""

    scores: tuple[float, ...]
    chains: tuple[int, ...]
    done_hands: int
    hmax: int
    beam_died: bool
    full: bool = False
    empty_cells: int = 0
    color_conflict: bool = False

    def at(self, hands: int) -> tuple[float, int, bool]:
        """総手数handsの(得点,連鎖数,打切り)。上限超は末尾値、満杯終了なら打切りFalse。"""
        h = max(1, min(int(hands), self.hmax))
        return self.scores[h - 1], self.chains[h - 1], bool(
            not self.full and (hands > self.hmax or (self.beam_died and h > self.done_hands)))

    def diagnostics(self, hands: int) -> dict[str, float]:
        """上限未達時だけ空きを記録。保持ビーム内の最大空き(隠し段込み)。"""
        truncated = self.at(hands)[2]
        empty = float(self.empty_cells) if truncated else float("nan")
        return {"trunc_empty": empty, "trunc_empty_nf": empty / BOARD_CAPACITY,
                "full": float(self.full and hands >= self.done_hands)}


def _norm_queue(queue: tuple[int, ...]) -> tuple[int, int, int, int]:
    q = tuple(int(v) if 1 <= int(v) <= 5 else 0 for v in queue)
    return (q + (0, 0, 0, 0))[:4]


@timed('search')
def ideal_fire_search(grid: np.ndarray, queue: tuple[int, ...], colors: tuple[int, ...] = ALL_COLORS,
                      hmax: int = HMAX_TOTAL_HANDS, resolve_before_death: bool = False,
                      beam_width: int = BEAM_WIDTH, potential_shortlist: int = POTENTIAL_SHORTLIST,
                      observed_colors: tuple[int, ...] = ()) -> IdealSearchResult:
    """H組以内の最大発火得点。shortlist=0は全候補、正数は幅に対する予備選抜倍率。

    colorsは使用可能色、observed_colorsは当時までに判明した色。
    4色未確定時は両者と両立する4色集合を経路ごとに保持する。
    潜在得点は選抜専用でQには加算しない。既存指標の探索は変更しない。
    resolve_before_deathは互換用引数。値によらず連鎖完了後に窒息を判定する。
    """
    arr = np.ascontiguousarray(grid, dtype=np.int8)
    if arr.shape != (BOARD_ROWS, BOARD_COLS):
        raise ValueError("盤面は13x6が必要")
    if (hmax < 1 or beam_width < 1 or potential_shortlist < 0 or not colors
            or any(c not in ALL_COLORS for c in (*colors, *observed_colors))):
        raise ValueError("手数・探索幅は正、探索色は有効な色が必要")
    return _search_cached(arr.tobytes(), _norm_queue(queue), tuple(sorted(set(colors))), min(int(hmax), HMAX_TOTAL_HANDS),
                          bool(resolve_before_death), int(beam_width), int(potential_shortlist), tuple(sorted(set(observed_colors))))


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _search_cached(raw: bytes, queue: tuple[int, int, int, int], colors: tuple[int, ...], hmax: int,
                   resolve: bool, beam: int, shortlist: int, observed: tuple[int, ...]) -> IdealSearchResult:
    grid = np.frombuffer(raw, dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    seen = set(observed) | {int(c) for c in np.unique(grid) if c in ALL_COLORS} | (set(queue) - {0})
    if len(colors) <= MATCH_COLOR_COUNT:
        seen.update(colors)
    if len(seen) > MATCH_COLOR_COUNT:
        # 色矛盾は物理的な満杯ではない。学習行は推定不能として欠測化する。
        return IdealSearchResult((0.0,) * hmax, (0,) * hmax, 0, hmax, True, False,
                                  int(np.count_nonzero(grid == 0)), True)
    result = native.b3_pair_search(grid, queue, colors, hmax, beam, resolve, shortlist, observed)
    scores, chains, done, died, full, empty = result
    return IdealSearchResult(tuple(float(s) for s in scores), tuple(chains), done, hmax, died, full, empty)


def search_manifest() -> dict[str, object]:
    """学習行に採用した探索設定を記録する。"""
    return dict(version=SEARCH_VERSION, beam_width=BEAM_WIDTH, hmax=HMAX_TOTAL_HANDS,
                potential_shortlist=POTENTIAL_SHORTLIST, hand_unit="pair", match_colors=MATCH_COLOR_COUNT,
                potential="one_drop_exact_score", hidden_row_pop=False,
                hidden_row_ojama_pop=False, death_check="after_chain")


def points_to_ojama(points: float, elapsed_first_move_sec: float) -> int:
    """得点→おじゃま個数。マージンレートは両側のうち早い最初の1手起点 (G20)。"""
    rate = compute_effective_rate(max(float(elapsed_first_move_sec), 0.0), from_first_move=True)
    return int(points // rate)


def clear_cache() -> None:
    _search_cached.cache_clear()
