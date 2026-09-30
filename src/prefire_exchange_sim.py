"""発火前の撃ち合い予測: 軽量2人対戦シミュレータ (2026-09-30、Phase 2、状態を持たない)。

user 決定 (2026-09-30) を固定規則とする:
- 規則として固定するのは「撃たないと窒息するなら撃つ」だけ (`forced`)。発火のタイミングは学習する
  (`src/prefire_exchange_layer.py` の hazard モデル、148動画)。
- 受け側は猶予手数内の最善応手を組むと仮定する (人の組み損ね率で割り引かない)。応手量は既存の
  近未来探索 `future_send` (S1′ の「打ち返しの余地」と同じ探索) の最大値を使う。
- 両者の NEXT/NEXT2 を使う。未読 (0 や 9 の番兵) なら探索しない (欠測)。

本モジュールは探索と特徴だけを返す。勝率への変換 (既存 S3 モデル) と混合は layer 側が行う。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src import puyo_core_bridge as native
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN, DEATH_COL, DEATH_ROW, Board
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import compute_effective_rate, score_to_ojama

PLAYABLE_COLORS = (1, 2, 3, 4, 5)
CELLS = BOARD_ROWS * BOARD_COLS                 # 78 (隠し段込み、飽和・埋まりの分母と統一)
SEND_NORMALIZER = 180.0                         # おじゃま個数の対数正規化の上端 (30段相当、最大級の大連鎖)
MAX_CHAIN_COUNT = 19                            # ぷよぷよの理論最大連鎖数
ELAPSED_NORMALIZER_SEC = 300.0                  # 経過秒の正規化上端 (マージンタイム後の典型試合長)
CACHE_SIZE = 65536
HAZARD_COLUMNS = (
    'own_send1', 'own_send2', 'own_chain1', 'own_chain2', 'own_can1', 'own_can2', 'own_forced',
    'opp_send1', 'opp_send2', 'opp_can1', 'opp_forced',
    'own_occupancy', 'opp_occupancy', 'own_ojama', 'opp_ojama',
    'own_death_col', 'opp_death_col', 'elapsed',
)


@dataclass(frozen=True)
class FireOption:
    """既知ツモでの1つの発火。placed は発火する組を置いた直後 (連鎖前) の盤面。"""

    hand: int
    score: int
    chain_count: int
    placed: bytes


@dataclass(frozen=True)
class SideFireOptions:
    """片側の既知2手以内の発火候補 (各手数で得点最大の1件) と、撃たないと窒息するか。"""

    best1: FireOption | None
    best2: FireOption | None
    forced: bool
    queue_known: bool

    def best(self) -> FireOption | None:
        """得点最大の発火 (同点なら早い手)。"""
        options = [o for o in (self.best1, self.best2) if o is not None]
        return max(options, key=lambda o: (o.score, -o.hand)) if options else None


NO_OPTIONS = SideFireOptions(None, None, False, False)


def queue_valid(queue: tuple[int, ...]) -> bool:
    """NEXT/NEXT2 の4色がすべて実色か (0 や 9 は未読の番兵)。"""
    return len(queue) == 4 and all(int(c) in PLAYABLE_COLORS for c in queue)


def _board(raw: bytes) -> Board:
    """int8 の生盤面から Board を作る。"""
    board = Board()
    board._grid = np.frombuffer(raw, dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS).astype(board._grid.dtype)
    return board


def _dead(grid: np.ndarray) -> bool:
    """窒息判定 (3列目の可視最上段、隠し段は含まない)。UNKNOWN は Board.is_dead と同じく窒息扱いしない。"""
    cell = int(grid[DEATH_ROW, DEATH_COL])
    return cell != 0 and cell != COLOR_UNKNOWN


def _placements(board: Board, pair: tuple[int, int]) -> list:
    """22配置を列挙して連鎖を解く (設置直後が窒息でも連鎖で消える可能性があるので除外しない)。"""
    return native.enumerate_and_simulate_placements(board, pair, filter_dead=False,
                                                    exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


def _as_option(hand: int, placement) -> FireOption:
    """配置結果を発火候補へ変換する。"""
    return FireOption(hand, int(placement.chain_result.exact_score), int(placement.chain_result.chain_count),
                      placement.placed_board._grid.astype(np.int8).tobytes())


def _split(placements: list) -> tuple[list, list]:
    """生き残る発火と、生き残る非発火に分ける。発火は連鎖後、非発火は設置直後で窒息を判定する。"""
    fires = [p for p in placements if p.chain_result.chain_count > 0
             and not _dead(p.chain_result.final_board._grid)]
    quiet = [p for p in placements if p.chain_result.chain_count == 0 and not _dead(p.placed_board._grid)]
    return fires, quiet


@lru_cache(maxsize=CACHE_SIZE)
def fire_options(raw: bytes, queue: tuple[int, ...]) -> SideFireOptions:
    """既知 NEXT/NEXT2 の2手を全列挙し、各手数の最大発火と「撃たないと窒息」を返す (純関数)。"""
    if not queue_valid(queue):
        return NO_OPTIONS
    first, second = (int(queue[0]), int(queue[1])), (int(queue[2]), int(queue[3]))
    fires1, quiet1 = _split(_placements(_board(raw), first))
    best1 = max(fires1, key=lambda p: p.chain_result.exact_score, default=None)
    best2 = None
    for placed in quiet1:
        fires2, _ = _split(_placements(placed.placed_board, second))
        top = max(fires2, key=lambda p: p.chain_result.exact_score, default=None)
        if top is not None and (best2 is None or top.chain_result.exact_score > best2.chain_result.exact_score):
            best2 = top
    forced = bool(fires1) and not quiet1   # 撃たないと窒息 = 生き残る非発火配置がない (user 規則)
    return SideFireOptions(_as_option(1, best1) if best1 is not None else None,
                           _as_option(2, best2) if best2 is not None else None, forced, True)


def send_ojama(score: float, elapsed_sec: float) -> float:
    """得点をおじゃま個数へ換算する (マージンタイム込み、端数は既存関数どおり切り捨て)。"""
    return float(score_to_ojama(int(score), elapsed_sec=elapsed_sec).ojama_count)


def _scaled_send(option: FireOption | None, elapsed_sec: float) -> float:
    """送り量を 0〜1 へ対数正規化する (候補なしは 0)。"""
    if option is None:
        return 0.0
    return float(min(1.0, np.log1p(send_ojama(option.score, elapsed_sec)) / np.log1p(SEND_NORMALIZER)))


def _board_stats(grid: np.ndarray) -> tuple[float, float, float]:
    """埋まり率・おじゃま率・窒息列の高さ (いずれも 0〜1)。"""
    filled = grid != 0
    heights = filled[:, DEATH_COL]
    column = float(BOARD_ROWS - int(np.argmax(heights))) if heights.any() else 0.0
    return float(filled.sum()) / CELLS, float((grid == COLOR_OJAMA).sum()) / CELLS, column / BOARD_ROWS


def hazard_features(own: SideFireOptions, opp: SideFireOptions, own_grid: np.ndarray,
                    opp_grid: np.ndarray, elapsed_sec: float) -> np.ndarray:
    """発火タイミング (hazard) モデルの入力。HAZARD_COLUMNS の順、すべて 0〜1。時刻 t 以前の観測だけ。"""
    own_stats, opp_stats = _board_stats(own_grid), _board_stats(opp_grid)
    chain = lambda o: 0.0 if o is None else min(1.0, o.chain_count / MAX_CHAIN_COUNT)
    values = [
        _scaled_send(own.best1, elapsed_sec), _scaled_send(own.best2, elapsed_sec),
        chain(own.best1), chain(own.best2), float(own.best1 is not None), float(own.best2 is not None),
        float(own.forced),
        _scaled_send(opp.best1, elapsed_sec), _scaled_send(opp.best2, elapsed_sec),
        float(opp.best1 is not None), float(opp.forced),
        own_stats[0], opp_stats[0], own_stats[1], opp_stats[1], own_stats[2], opp_stats[2],
        min(1.0, max(0.0, elapsed_sec) / ELAPSED_NORMALIZER_SEC),
    ]
    return np.asarray(values, dtype=float)


def effective_rate(elapsed_sec: float) -> int:
    """おじゃま1個あたりの得点 (マージンタイム込み)。応手量を得点へ戻すときに使う。"""
    return int(compute_effective_rate(elapsed_sec))
