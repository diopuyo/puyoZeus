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
    board_stable: bool = True

    def best(self) -> FireOption | None:
        """得点最大の発火 (同点なら早い手)。"""
        options = [o for o in (self.best1, self.best2) if o is not None]
        return max(options, key=lambda o: (o.score, -o.hand)) if options else None


NO_OPTIONS = SideFireOptions(None, None, False, False)
UNSTABLE_BOARD = SideFireOptions(None, None, False, True, board_stable=False)


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


def _split(placements: list) -> tuple[list, list, bool]:
    """生き残る発火・生き残る非発火・非発火の配置が1つでもあるか。発火は連鎖後、非発火は設置直後で窒息を判定する。"""
    fires = [p for p in placements if p.chain_result.chain_count > 0
             and not _dead(p.chain_result.final_board._grid)]
    quiet_any = [p for p in placements if p.chain_result.chain_count == 0]
    quiet = [p for p in quiet_any if not _dead(p.placed_board._grid)]
    return fires, quiet, bool(quiet_any)


@lru_cache(maxsize=CACHE_SIZE)
def fire_options(raw: bytes, queue: tuple[int, ...]) -> SideFireOptions:
    """既知 NEXT/NEXT2 の2手を全列挙し、各手数の最大発火と「撃たないと窒息」を返す (純関数)。"""
    if not queue_valid(queue):
        return NO_OPTIONS
    first, second = (int(queue[0]), int(queue[1])), (int(queue[2]), int(queue[3]))
    board = _board(raw)
    if _dead(board._grid):
        return UNSTABLE_BOARD   # 置く前から窒息セルが埋まっている盤面 (誤読・終局) は探索しない
    if native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).chain_count > 0:
        return UNSTABLE_BOARD   # 置く前から消える群がある盤面は STABLE ではない (誤読・連鎖途中)。探索しない
    fires1, quiet1, has_quiet = _split(_placements(board, first))
    best1 = max(fires1, key=lambda p: p.chain_result.exact_score, default=None)
    best2 = None
    for placed in quiet1:
        fires2, _, _ = _split(_placements(placed.placed_board, second))
        top = max(fires2, key=lambda p: p.chain_result.exact_score, default=None)
        if top is not None and (best2 is None or top.chain_result.exact_score > best2.chain_result.exact_score):
            best2 = top
    # 撃たないと窒息 (user 規則) = 撃たない置き方はあるが全部窒息し、生き残る発火がある。
    # どこに置いても消える盤面は「選んで撃つ」ではないので含めない。
    forced = bool(fires1) and has_quiet and not quiet1
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


COUNTER_ROLLOUTS = 8          # 受け側の応手を見積もる未来ツモの標本数 (NEXT/NEXT2 より先は未知)
COUNTER_BEAM_WIDTH = 8        # 応手探索のビーム幅 (既存の近未来火力 NEAR_FUTURE_BEAM_WIDTH と同じ)
MAX_COUNTER_HANDS = 16        # 応手探索の深さ上限 (最大級の連鎖演出でも着弾までに置ける手数の上限)


def _rollout_pairs(rng: np.random.Generator, known: tuple[int, ...], hands: int,
                   colors: tuple[int, ...]) -> list[tuple[int, int]]:
    """既知の NEXT/NEXT2 の後ろへ、試合で見えた色から等確率の仮ツモを足す (1標本分)。"""
    pairs = []
    for i in range(0, len(known) - 1, 2):   # 見えている組は先頭から、最初の未読で打ち切る
        if not (int(known[i]) in PLAYABLE_COLORS and int(known[i + 1]) in PLAYABLE_COLORS):
            break
        pairs.append((int(known[i]), int(known[i + 1])))
    pairs = pairs[:hands]
    while len(pairs) < hands:
        pairs.append((int(rng.choice(colors)), int(rng.choice(colors))))
    return pairs


def counter_scores(raw: bytes, queue: tuple[int, ...], hands: int, colors: tuple[int, ...],
                   seed: int) -> np.ndarray:
    """受け側が hands 手以内に撃てる最大得点を、未来ツモの標本ごとに返す (最善応手・割り引きなし)。

    未知のツモは人のミスではなく運なので標本で扱う (user 決定 9/30: 探索の不確かさだけを扱う)。
    NEXT が未読・盤面が不正なら 0 点 (応手なしの欠測、楽観で埋めない)。seed は盤面から決まる (再現可能)。
    """
    if not queue_valid(tuple(queue[:4])) or not colors or fire_options(raw, tuple(queue[:4])) is UNSTABLE_BOARD:
        return np.zeros(COUNTER_ROLLOUTS)
    board, depth = _board(raw), int(min(max(1, hands), MAX_COUNTER_HANDS))
    rng = np.random.default_rng(seed)
    out = np.zeros(COUNTER_ROLLOUTS)
    for i in range(COUNTER_ROLLOUTS):
        result = native.beam_search(board, _rollout_pairs(rng, queue, depth, colors), COUNTER_BEAM_WIDTH,
                                    exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED, use_exact_score=True)
        out[i] = float(result.best_score)
    return out


def seen_colors(grids: list[np.ndarray], queues: list[tuple[int, ...]]) -> tuple[int, ...]:
    """時刻 t までに盤面と NEXT で見えた色 (試合は4色、memory reference_four_colors_per_match)。"""
    values = {int(v) for g in grids for v in np.unique(g)} | {int(v) for q in queues for v in q}
    return tuple(sorted(v for v in values if v in PLAYABLE_COLORS))


def stable_seed(*parts: bytes | tuple | int) -> int:
    """盤面・NEXT・手数から決まる乱数の種 (同じ入力なら同じ標本)。"""
    import hashlib
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], 'little')
