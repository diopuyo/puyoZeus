"""CounterReachTracker の時間ベース再計算間引き (2026-08-12 追加) の回帰テスト。

背景: `_reach` は1回0.35〜数秒かかる MC 計算で、打ち合い場面では相手盤面が
毎フレーム変わるためキャッシュキーが変わり続け、間引き無しでは毎秒複数回
走っていた (デモ生成が遅い主犯)。本テストは重い実 MC を走らせず
`CounterReachTracker._reach` をモックして呼び出し回数のみ検証する
(既存 `tests/test_counter_reach_probability.py` は MC 本体の契約テストで
本ファイルとは責務が異なる、変更なし)。
"""
from __future__ import annotations

import math

from scripts.visualize_advantage_overlay import (
    COUNTER_RECOMPUTE_INTERVAL_SEC,
    CounterReachTracker,
)
from src.board import Board, COLOR_RED, COLOR_BLUE


def _board_with(row: int, col: int, color: int) -> Board:
    """指定セルのみ埋めた最小盤面 (呼び出しごとに内容を変えてキャッシュキーを変える用)。"""
    board = Board()
    board.set(row, col, color)
    return board


def _tracker_with_stub_reach() -> tuple[CounterReachTracker, dict]:
    """_reach をスタブ化した tracker と呼び出し回数カウンタを返す。"""
    tracker = CounterReachTracker()
    counts = {"n": 0}

    def _stub_reach(board: Board, budget_sec: float, known_pairs: tuple) -> tuple[float, float]:
        counts["n"] += 1
        return 0.5, 2.0

    tracker._reach = _stub_reach  # type: ignore[method-assign]
    return tracker, counts


def test_throttle_skips_recompute_within_interval() -> None:
    """0.5秒未満の間隔かつ盤面が変わっても、前回結果を再利用し _reach は呼ばない。"""
    tracker, counts = _tracker_with_stub_reach()
    b1a = _board_with(12, 0, COLOR_RED)
    b2a = _board_with(12, 5, COLOR_BLUE)
    r1 = tracker.update(b1a, b2a, budget_sec=1.0, t_sec=1.0)
    assert counts["n"] == 2  # 1P/2P各1回

    # 盤面を変えても (=旧dictキャッシュならmiss) 0.2秒後は再利用されるはず。
    b1b = _board_with(12, 1, COLOR_RED)
    b2b = _board_with(12, 4, COLOR_BLUE)
    r2 = tracker.update(b1b, b2b, budget_sec=1.0, t_sec=1.0 + 0.2)
    assert counts["n"] == 2, "間引き区間内なのに再計算された"
    assert r2 == r1


def test_recompute_after_interval_elapsed() -> None:
    """COUNTER_RECOMPUTE_INTERVAL_SEC 以上経過したら再計算される。

    2回目は盤面内容も変える (同一盤面+同一budgetだと旧dictキャッシュが
    ヒットしてしまい `_reach` 呼び出し有無だけでは時間間引きの検証にならない
    ため)。
    """
    tracker, counts = _tracker_with_stub_reach()
    b1a = _board_with(12, 0, COLOR_RED)
    b2a = _board_with(12, 5, COLOR_BLUE)
    tracker.update(b1a, b2a, budget_sec=1.0, t_sec=0.0)
    assert counts["n"] == 2

    b1b = _board_with(12, 1, COLOR_RED)
    b2b = _board_with(12, 4, COLOR_BLUE)
    t_next = COUNTER_RECOMPUTE_INTERVAL_SEC + 0.01
    tracker.update(b1b, b2b, budget_sec=1.0, t_sec=t_next)
    assert counts["n"] == 4, "間引き間隔経過後に再計算されなかった"


def test_budget_transition_forces_immediate_recompute() -> None:
    """budget が 0→正 に変わった直後は間引きを無視して即計算する (反応遅れ防止)。"""
    tracker, counts = _tracker_with_stub_reach()
    b1 = _board_with(12, 0, COLOR_RED)
    b2 = _board_with(12, 5, COLOR_BLUE)

    # 打ち合い開始前 (budget=0): 計算されず即0を返す。
    r0 = tracker.update(b1, b2, budget_sec=0.0, t_sec=0.0)
    assert r0[0] == 0.0 and math.isnan(r0[1]) and math.isnan(r0[2])
    assert counts["n"] == 0

    # ごく短時間後 (0.01秒、通常なら間引き区間内) に budget が正へ遷移
    # → 遷移直後なので間引きを無視して即計算されるはず。
    tracker.update(b1, b2, budget_sec=1.0, t_sec=0.01)
    assert counts["n"] == 2, "budget 遷移直後なのに間引きされた"


def test_t_sec_omitted_keeps_legacy_no_throttle_behavior() -> None:
    """t_sec 省略時は従来通り毎回計算する (後方互換)。"""
    tracker, counts = _tracker_with_stub_reach()
    b1a = _board_with(12, 0, COLOR_RED)
    b2a = _board_with(12, 5, COLOR_BLUE)
    b1b = _board_with(12, 1, COLOR_RED)
    b2b = _board_with(12, 4, COLOR_BLUE)

    tracker.update(b1a, b2a, budget_sec=1.0)
    tracker.update(b1b, b2b, budget_sec=1.0)
    assert counts["n"] == 4, "t_sec 省略時に間引きが誤って発動した"
