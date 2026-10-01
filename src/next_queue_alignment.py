"""NEXT/DNEXT の読みを「次に置く組 P_k, その次 P_{k+1}」の意味へ揃える (2026-10-01)。

背景 (exev logs/next_shift/RESULT.md):
- 通常設置の直後は STABLE 区間の先頭で next = P_k (繰り上がり前)。
- 連鎖後・おじゃま後は STABLE 開始時点で既に繰り上がり済みで next = P_{k+1}、
  P_k は手中の組 (画面の盤面上部に表示) で NEXT 枠から消えている。

ここでは状態を持たない関数だけを置く (行単位の学習データ補正用)。
提供側の逐次整列器 (状態あり) は src/next_queue_serving_alignment.py の外部ラッパーが持つ。
"""
from __future__ import annotations

import numpy as np

COLOR_MIN, COLOR_MAX = 1, 5
OJAMA = 9
PAIR_CELLS = 2
QUEUE_SIZE = 4
KIND_PLACE, KIND_CHAIN, KIND_OJAMA = "place", "chain", "ojama"
KIND_MULTI, KIND_SAME, KIND_START = "multi", "same", "start"
# 行補正の方式コード (学習行の監査用)
METHOD_KEEP, METHOD_SHIFT, METHOD_INVALID = 0, 1, 2
# 真値選択で「t 以前に表示された組」とみなす範囲 (今の行 + 直前 N 行の next/dnext)。
# 連鎖後の手中の組は直前の連鎖中の行の next に出る (logs/next_shift_train/measure.json)。
TRUTH_LOOKBACK = 2


def valid_pair(pair: tuple) -> bool:
    """2色とも色ぷよ (1〜5) なら True。"""
    return len(pair) == PAIR_CELLS and all(COLOR_MIN <= int(v) <= COLOR_MAX for v in pair)


def pair_key(a: int, b: int) -> tuple[int, int]:
    """無順序の組のキー。"""
    return (min(int(a), int(b)), max(int(a), int(b)))


def transition_kind(prev: np.ndarray, cur: np.ndarray) -> tuple[str, tuple[int, int] | None]:
    """盤面 prev→cur の遷移分類と、きれいな設置ならその組 (無順序)。"""
    added = (prev == 0) & (cur != 0)
    removed = (prev != 0) & (cur == 0)
    changed = (prev != 0) & (cur != 0) & (prev != cur)
    if removed.any() or changed.any():
        return KIND_CHAIN, None
    values = cur[added]
    if len(values) == PAIR_CELLS and ((values >= COLOR_MIN) & (values <= COLOR_MAX)).all():
        return KIND_PLACE, pair_key(int(values[0]), int(values[1]))
    if len(values) and (values == OJAMA).any():
        return KIND_OJAMA, None
    return (KIND_SAME, None) if not len(values) else (KIND_MULTI, None)


def entry_kinds(grids: np.ndarray) -> list[str]:
    """各行へ入った遷移の種類 (先頭行は start)。盤面は過去と現在のみ使う。"""
    return [KIND_START] + [transition_kind(grids[i - 1], grids[i])[0] for i in range(1, len(grids))]


def placed_pairs(grids: np.ndarray) -> list[tuple[int, int] | None]:
    """各行の直後に置かれた組 (次の行とのきれいな設置差分)。未来を使うので検証専用。"""
    out: list[tuple[int, int] | None] = [transition_kind(grids[i], grids[i + 1])[1]
                                         for i in range(len(grids) - 1)]
    return out + [None]


def _pairs(q: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """queue4 → (next, dnext) の順序付き組。"""
    return (int(q[0]), int(q[1])), (int(q[2]), int(q[3]))


def is_shifted(base: np.ndarray, cur: np.ndarray) -> bool:
    """cur が base から1手繰り上がった読みか (cur.next == base.dnext かつ全体は不一致)。"""
    (b1, b2), (c1, c2) = _pairs(base), _pairs(cur)
    if not (valid_pair(b1) and valid_pair(b2) and valid_pair(c1)):
        return False
    same = pair_key(*b1) == pair_key(*c1) and pair_key(*b2) == pair_key(*c2)
    return pair_key(*c1) == pair_key(*b2) and not same


def causal_row_queue(entry: str, prev_queue: np.ndarray | None,
                     queue: np.ndarray) -> tuple[np.ndarray, int]:
    """行 r の queue を (P_k, P_{k+1}) の意味へ揃える。入力は行 r 以前の読みだけ。

    通常設置の後 (entry=place) と試合開始は区間先頭の読みがそのまま P_k。
    それ以外 (連鎖・おじゃま・複数手の後) で、直前行の読みから1手繰り上がっていれば
    手中の組 = 直前行の next を先頭へ戻し (直前 next, 今の next) にする。
    """
    q = np.asarray(queue, dtype=np.int8)
    if not all(valid_pair(p) for p in _pairs(q)):
        return q.copy(), METHOD_INVALID
    if entry in (KIND_PLACE, KIND_START) or prev_queue is None:
        return q.copy(), METHOD_KEEP
    base = np.asarray(prev_queue, dtype=np.int8)
    if not is_shifted(base, q):
        return q.copy(), METHOD_KEEP
    return np.asarray((*base[:PAIR_CELLS], *q[:PAIR_CELLS]), dtype=np.int8), METHOD_SHIFT


def displayed_slots(queues: np.ndarray, r: int, lookback: int) -> list[tuple[int, int]]:
    """行 r から lookback 行前までに表示枠で読めた組 (順序付き、有効色のみ)。t 以前の表示だけ。"""
    out: list[tuple[int, int]] = []
    for i in range(r, max(-1, r - lookback - 1), -1):
        out.extend(p for p in _pairs(queues[i]) if valid_pair(p))
    return out


def _select(truth: tuple[int, int] | None, shown: list[tuple[int, int]]) -> tuple[int, int] | None:
    """真の組が表示枠に出ていれば、その表示どおりの順序で返す (出ていなければ None)。"""
    if truth is None:
        return None
    return next((p for p in shown if pair_key(*p) == truth), None)


def truth_row_queue(r: int, grids: np.ndarray, queues: np.ndarray,
                    fallback: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """行 r の (P_k, P_{k+1}) を、実際に置かれた組で表示枠から選ぶ。

    使うのは 行 r 以前の表示の読み (queues[:r+1]) と、選択のための盤面 grids[r:r+3] だけ。
    出力の色は必ず t 以前に表示枠で読めた組のどれか (表示されていない情報は入らない)。
    選べない枠は fallback (因果規則の出力) を使う。戻り値の2番目は枠ごとの由来 (1=真値選択, 0=fallback)。
    """
    pk = transition_kind(grids[r], grids[r + 1])[1] if r + 1 < len(grids) else None
    pk1 = (transition_kind(grids[r + 1], grids[r + 2])[1]
           if pk is not None and r + 2 < len(grids) else None)
    first = _select(pk, displayed_slots(queues, r, TRUTH_LOOKBACK))
    second = _select(pk1, displayed_slots(queues, r, TRUTH_LOOKBACK))
    out = np.array(fallback, dtype=np.int8, copy=True)
    if first is not None:
        out[:PAIR_CELLS] = first
    if second is not None:
        out[PAIR_CELLS:] = second
    return out, (int(first is not None), int(second is not None))


def truth_sequence(grids: np.ndarray, queues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """真値選択版。戻り値 = (補正queue, 因果規則の方式コード, 枠ごとの真値選択フラグ(行,2))。"""
    causal, methods = correct_sequence(grids, queues)
    out = causal.copy()
    chosen = np.zeros((len(grids), PAIR_CELLS), dtype=np.int8)
    for r in range(len(grids)):
        out[r], chosen[r] = truth_row_queue(r, grids, queues, causal[r])
    return out, methods, chosen


def correct_sequence(grids: np.ndarray, queues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1側1試合の時刻順の行列を補正する。行 r の出力は行 0..r の入力だけで決まる。"""
    entries = entry_kinds(grids)
    out = np.array(queues, dtype=np.int8, copy=True)
    methods = np.zeros(len(grids), dtype=np.int8)
    for r in range(len(grids)):
        prev = queues[r - 1] if r else None
        out[r], methods[r] = causal_row_queue(entries[r], prev, queues[r])
    return out, methods
