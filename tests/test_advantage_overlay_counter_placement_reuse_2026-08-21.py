"""応手MCの「設置ごと量子化」(2026-08-21 user承認) のテスト。

対象: `CounterReachTracker._update_defender_only` の `reuse_if_board_unchanged`
(既定 False)。受け側の盤面bytesが前回計算時と同一なら、時間予算 (budget_sec)
が変化していても再計算せず前回の (応手確率, 平均打手数) を再利用する。
閾値 (threshold_ojama) が変わればスコープが別物になり自動的に再計算される。

`mc_counter_estimator.estimate_counter_distribution` (実際のモンテカルロ、
重い) は呼ばず、`CounterReachTracker._reach` を軽量スタブに monkeypatch して
「実際に再計算が起きたか」を呼出回数で直接検証する
(measured hit-rate は別途 scripts 配下の一時計測スクリプトで実クリップに対し
0.00% と実測済み、本ファイルはロジックの単体テストに専念する)。
"""
from __future__ import annotations

import inspect

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board
from src.board import HIDDEN_ROWS
import scripts.visualize_advantage_overlay as vao


def _board_with_ojama(n: int) -> Board:
    """可視領域下段から n 個のお邪魔を敷いた Board (test_advantage_components.py
    の同名ヘルパーと同一設計、盤面bytesを変えるためだけに使う)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    placed = 0
    for row in range(BOARD_ROWS - 1, HIDDEN_ROWS - 1, -1):
        for col in range(BOARD_COLS):
            if placed >= n:
                break
            grid[row][col] = COLOR_OJAMA
            placed += 1
        if placed >= n:
            break
    return Board.from_list(grid)


def _patch_reach_with_counter(monkeypatch: pytest.MonkeyPatch) -> dict:
    """`_reach` を軽量スタブに差し替え、呼出回数を計数する。

    戻り値は呼出ごとに単調増加する (p, h) を返すため、「2回目が1回目と同じ
    値なら再利用された」ことをアサートできる。
    """
    calls = {"n": 0}

    def fake_reach(self, board, budget_sec, known_pairs,
                   threshold_ojama: float = vao.COUNTER_THRESHOLD_OJAMA) -> tuple[float, float]:
        calls["n"] += 1
        return (float(calls["n"]), float(calls["n"]) * 10.0)

    monkeypatch.setattr(vao.CounterReachTracker, "_reach", fake_reach)
    return calls


# ============================
# CounterReachTracker._update_defender_only / update() のロジック
# ============================


def test_reuse_off_by_default_recomputes_every_call_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既定 (reuse_if_board_unchanged 省略=False) は盤面不変でも毎回再計算する
    (従来挙動、backwards compat)。`_placement_reuse` も一切書き込まれない。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    tracker.update(board, board, 5.0, defender_side="1P", threshold_ojama=6.0)
    tracker.update(board, board, 4.5, defender_side="1P", threshold_ojama=6.0)
    assert calls["n"] == 2
    assert tracker._placement_reuse == {}


def test_reuse_on_skips_recompute_when_board_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reuse_if_board_unchanged=True かつ盤面bytesが同一なら、budget_sec が
    変化していても2回目は再計算せず前回の (応手確率, 平均打手数) を返す。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    _, p1_first, _ = tracker.update(
        board, board, 5.0, defender_side="1P", threshold_ojama=6.0,
        reuse_if_board_unchanged=True)
    _, p1_second, _ = tracker.update(
        board, board, 4.5, defender_side="1P", threshold_ojama=6.0,
        reuse_if_board_unchanged=True)
    assert calls["n"] == 1  # 2回目は _reach を呼ばない
    assert p1_first == p1_second  # 前回値をそのまま再利用


def test_reuse_on_recomputes_when_board_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """盤面bytesが変わった (=設置が起きた) 場合は reuse_if_board_unchanged=True
    でも再計算する。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board_a = _board_with_ojama(0)
    board_b = _board_with_ojama(3)
    tracker.update(board_a, board_a, 5.0, defender_side="1P", threshold_ojama=6.0,
                    reuse_if_board_unchanged=True)
    tracker.update(board_b, board_b, 4.5, defender_side="1P", threshold_ojama=6.0,
                    reuse_if_board_unchanged=True)
    assert calls["n"] == 2


def test_reuse_on_recomputes_when_threshold_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """盤面bytesが同一でも閾値 (threshold_ojama) が変わればスコープが別物に
    なり再計算する (「閾値が違えば再計算が必要」指摘の反映)。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    tracker.update(board, board, 5.0, defender_side="1P", threshold_ojama=6.0,
                    reuse_if_board_unchanged=True)
    tracker.update(board, board, 4.5, defender_side="1P", threshold_ojama=12.0,
                    reuse_if_board_unchanged=True)
    assert calls["n"] == 2


def test_reuse_on_scopes_are_independent_per_defender_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1P/2P はスコープが別物 (お互いのキャッシュを汚染しない)。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    tracker.update(board, board, 5.0, defender_side="1P", threshold_ojama=6.0,
                    reuse_if_board_unchanged=True)
    tracker.update(board, board, 5.0, defender_side="2P", threshold_ojama=6.0,
                    reuse_if_board_unchanged=True)
    assert calls["n"] == 2  # 1P用の再利用は2P側には効かない


# ============================
# 配線の署名・既定値・静的回帰テスト
# ============================


def test_counter_reach_tracker_update_signature_new_arg_default_false() -> None:
    sig = inspect.signature(vao.CounterReachTracker.update)
    assert sig.parameters["reuse_if_board_unchanged"].default is False


def test_resolved_exchange_tracker_counter_placement_reuse_default_false() -> None:
    sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    assert sig.parameters["enable_counter_placement_reuse"].default is False
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_counter_placement_reuse is False


def test_generate_signature_has_enable_resolved_counter_placement_reuse_default_false() -> None:
    sig = inspect.signature(vao.generate)
    assert "enable_resolved_counter_placement_reuse" in sig.parameters
    assert sig.parameters["enable_resolved_counter_placement_reuse"].default is False


def test_cli_resolved_counter_placement_reuse_flag_defaults_to_false() -> None:
    src = inspect.getsource(vao.main)
    assert '"--resolved-counter-placement-reuse"' in src
    assert 'dest="enable_resolved_counter_placement_reuse"' in src


def test_generate_source_wires_counter_placement_reuse_to_both_constructions() -> None:
    """静的回帰テスト: generate() ソース中の ResolvedExchangeTracker 構築
    (通常時/試合境界リセット時の2箇所) が両方とも
    enable_counter_placement_reuse=enable_resolved_counter_placement_reuse を
    渡していることを固定する (配線漏れ防止)。"""
    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    pattern = "enable_counter_placement_reuse=enable_resolved_counter_placement_reuse"
    assert code_only.count(pattern) == 2


def test_resolved_exchange_tracker_source_wires_reuse_flag_to_both_counter_calls() -> None:
    """静的回帰テスト: ResolvedExchangeTracker 内の counter_tracker.update 呼出
    2箇所 (_amplify_decisive / _reevaluate_live_defender) が両方とも
    reuse_if_board_unchanged=self._enable_counter_placement_reuse を渡す。"""
    src = inspect.getsource(vao.ResolvedExchangeTracker)
    assert src.count("reuse_if_board_unchanged=self._enable_counter_placement_reuse") == 2
