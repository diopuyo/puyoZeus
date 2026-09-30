"""cycle65「NEXT履歴の色補正」の対整合ガード (既定OFF、純粋関数のみ)。

背景 (2026-09-30, logs/match14_misread/DIAGNOSIS.md): 置いた手の確定がおじゃま落下などで
遅れると、キュー末尾側がすでに 1 手先へ進んでおり、観測色と合わない対で新規 2 セルを
上書きしてしまう (q 全 24 回中 21 回が不一致・書換 34 セルで正解 0)。
このモジュールは「観測した新規 2 セルの色」と整合する対だけを採用する判定を提供する。
"""
from __future__ import annotations

from typing import Sequence

# 色定数の重複を避けるため board の値をそのまま使う (0=空, 9=おじゃま, 10=不明)。
COLOR_EMPTY = 0
COLOR_OJAMA = 9
COLOR_UNKNOWN = 10

# 整合する対を探すキュー履歴の件数 (末尾から数えた直近ペア数)。
RECOLOR_HISTORY_LOOKBACK = 3

# 監査カウンタのキー。
OUTCOME_ACCEPTED = "accepted"          # 従来の対が観測と整合 → 従来どおり補正
OUTCOME_SUBSTITUTED = "substituted"    # 履歴内の別の対に差し替えて補正
OUTCOME_KEPT_OBSERVED = "kept_observed"  # 整合する対なし → 観測色のまま (補正しない)
OUTCOMES = (OUTCOME_ACCEPTED, OUTCOME_SUBSTITUTED, OUTCOME_KEPT_OBSERVED)

Pair = tuple[int, int]


def is_usable_pair(pair: Pair | None) -> bool:
    """補正に使える対か (空・おじゃま・不明を含まない)。"""
    return pair is not None and all(
        c not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN) for c in pair
    )


def pair_consistent(observed: Sequence[int], pair: Pair) -> bool:
    """観測色 (不明を除く) が対の多重集合に含まれるか。

    おじゃま観測は無視しない: 実測で cycle65 発火時の観測におじゃまを含む 9 呼出のうち
    8 件は実際におじゃま (真値=観測) で、無視すると本物のおじゃまを色で潰す。
    誤読の 1 件 (zenchi 3155.75秒 1P) は取りこぼす (logs/c65_guard/RESULT.md)。
    """
    remaining = list(pair)
    for color in observed:
        if color == COLOR_UNKNOWN:
            continue
        if color not in remaining:
            return False
        remaining.remove(color)
    return True


def _candidate_indices(length: int, lookback: int) -> list[int]:
    """従来の採用位置 → それより古い履歴 (新しい順) → 新しい側、の探索順を返す。"""
    base = length - 2 if length >= 2 else length - 1
    oldest = max(0, length - lookback)
    older = list(range(base - 1, oldest - 1, -1))
    newer = list(range(base + 1, length))
    return [base] + older + newer


def select_recolor_pair(
    observed: Sequence[int], queue: Sequence[Pair],
    lookback: int = RECOLOR_HISTORY_LOOKBACK,
) -> tuple[Pair | None, str]:
    """観測色と整合する補正用の対を選ぶ。(対 or None, 結果種別) を返す。

    queue が空なら (None, kept_observed)。従来の採用対が整合すれば accepted、
    直近 lookback 件のうち整合する対があれば substituted、無ければ kept_observed。
    """
    if not queue:
        return None, OUTCOME_KEPT_OBSERVED
    for order, index in enumerate(_candidate_indices(len(queue), lookback)):
        pair = queue[index]
        if is_usable_pair(pair) and pair_consistent(observed, pair):
            return pair, OUTCOME_ACCEPTED if order == 0 else OUTCOME_SUBSTITUTED
    return None, OUTCOME_KEPT_OBSERVED
