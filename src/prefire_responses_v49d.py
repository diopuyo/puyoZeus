"""連続値用の応手列挙 (v4.9d)。実際に置ける手と、その手自身が生む得点の組を返す。

証明用の拡大集合 (全5色・他の枝の最大相殺を無償付与) は使わない。打切りは作業量 (ノード数) と
手数だけで決め、壁時計に依存しない。打切り時は完了した段までの結果を返す (段の途中は捨てる)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
from typing import Iterable
import numpy as np
from src.board import Board, BOARD_COLS
from src.exchange_event_multilanding import Search, SearchCutoff, PAIR_SIZE
from src.exchange_fast_expand import drop_ojama_fast
from src.exchange_multilanding_precheck import placement_count
from src.scoring import OJAMA_MAX_DROP_PER_TURN, score_to_ojama

# 応手列挙のノード上限 (1配置 = 1ノード)。壁時計ではなく作業量で止める。
RESPONSE_NODE_LIMIT = 3000
# 応手として読む手数の上限 (観測できる着弾までの手数と、この値の小さい方)。
RESPONSE_MAX_HANDS = 2
# 端数列の置き場所は、辞書順の中央の1通りで代表させる (B3評価の回数を抑える)。
ARRANGEMENT_SAMPLES = 1


@dataclass
class ResponseSearch(Search):
    """試合で使われる4色だけで未知ツモを展開する応手列挙器。"""
    colors: tuple = (1, 2, 3, 4)
    stage: dict = field(default_factory=dict)

    def pairs_for(self, queue: tuple, hand: int) -> list[tuple]:
        pair = queue[PAIR_SIZE*hand:PAIR_SIZE*(hand+1)]
        known = len(pair) == PAIR_SIZE and all(v in (1, 2, 3, 4, 5) for v in pair)
        return [tuple(pair)] if known else list(combinations_with_replacement(self.colors, PAIR_SIZE))


# 出力を変えない高速化 (全行一致を確認済みの場合だけ有効): 応手の列挙は盤面・ツモ・手数・色だけで決まるため、
# 予告の量だけが変わる更新では結果を再利用する。False で従来の毎回計算。
EXACT_RESPONSE_MEMO = True
RESPONSE_MEMO_LIMIT = 1024
_RESPONSE_MEMO: dict[tuple, dict] = {}


def feasible_responses(board: Board, queue: tuple, hands: int, elapsed: float,
                       colors: tuple, limit: int = RESPONSE_NODE_LIMIT) -> dict:
    """手数 `hands` までの全配置を、連鎖解決後の盤面と累計得点の組で返す (窒息する手は除く)。

    同じ盤面へ至る手順が複数ある場合は、得点の最大を残す (相殺量を過小に見ない)。
    返り値の replies は (盤面, 累計得点) のリスト。complete は最後まで読めたか。
    """
    if not EXACT_RESPONSE_MEMO:
        return enumerate_responses(board, queue, hands, elapsed, colors, limit)
    key = (board._grid.tobytes(), tuple(queue), int(hands), tuple(colors), int(limit))
    if key not in _RESPONSE_MEMO:
        if len(_RESPONSE_MEMO) >= RESPONSE_MEMO_LIMIT:
            _RESPONSE_MEMO.pop(next(iter(_RESPONSE_MEMO)))
        _RESPONSE_MEMO[key] = enumerate_responses(board, queue, hands, elapsed, colors, limit)
    return _RESPONSE_MEMO[key]


def enumerate_responses(board: Board, queue: tuple, hands: int, elapsed: float,
                        colors: tuple, limit: int) -> dict:
    """応手の列挙本体 (ノード上限は作業量。壁時計は使わない)。"""
    search = ResponseSearch(lambda *args: 0., elapsed, limit, colors=tuple(colors))
    frontier, replies, done = [(board, 0)], {}, 0
    try:
        for hand in range(max(0, min(hands, RESPONSE_MAX_HANDS))):
            pairs = search.pairs_for(queue, hand)
            search.reserve(len(pairs)*sum(placement_count(b._grid) for b, _ in frontier))
            level: dict = {}
            for current, points in frontier:
                for final, score in search.outcomes(current, pairs):
                    if final.is_dead():
                        continue
                    key = final._grid.tobytes()
                    previous = level.get(key, (None, -1))[1]
                    level[key] = (final, max(previous, points+int(score)))
            frontier = list(level.values())
            for key, (final, points) in level.items():
                replies[key] = (final, max(points, replies.get(key, (None, -1))[1]))
            done = hand+1
    except SearchCutoff:
        return dict(replies=list(replies.values()), complete=False, nodes=search.nodes, hands_done=done)
    return dict(replies=list(replies.values()), complete=True, nodes=search.nodes, hands_done=done)


def cancel_amount(points: int, carry: int, elapsed: float) -> int:
    """応手の得点が相殺する個数。受け側の端数繰越を台帳と同じに換算する。"""
    return int(score_to_ojama(int(points), prev_leftover=int(carry), elapsed_sec=elapsed).ojama_count)


def arrangement_columns(dropped: int) -> list[tuple[int, ...]]:
    """端数の置き場所の代表。端数0なら1通り、それ以外は辞書順の中央の1通り。"""
    combos = list(combinations(range(BOARD_COLS), dropped % BOARD_COLS))
    return combos[len(combos)//2:len(combos)//2+ARRANGEMENT_SAMPLES]


def landed_boards(board: Board, dropped: int) -> list[Board]:
    """1回の着弾 (`dropped` 個) を代表の置き場所で落とした盤面。"""
    if dropped <= 0:
        return [board]
    return [drop_ojama_fast(board, dropped, columns) for columns in arrangement_columns(dropped)]


def landing_count(pending: int) -> int:
    """1回の着弾の個数 (上限30)。"""
    return min(max(0, int(pending)), OJAMA_MAX_DROP_PER_TURN)


def representative(replies: Iterable[tuple[Board, int]]) -> list[tuple[Board, int]]:
    """決定的な順序: 得点の大きい順、空きの多い順 (ぷよの少ない順)、盤面バイト列。"""
    return sorted(replies, key=lambda r: (-r[1], int(np.count_nonzero(r[0]._grid)), r[0]._grid.tobytes()))
