"""応手MCの「残り秒数の量子化」(2026-08-21 user承認) のテスト。

対象: `CounterReachTracker._update_defender_only` の `quantize_budget_sec`
(既定 False)。キャッシュキーに入る budget_sec (着弾までの残り秒) を
`COUNTER_BUDGET_QUANTUM_SEC` (1手あたりの平均設置時間、
`mc_counter_estimator.PLACEMENT_SPEED_BY_ROW_SEC` の単純平均、物理実測値
からの導出) 単位に丸める。同一の量子化バケットに入る budget_sec は
「打てる手数が同じ=答えが同じ」という近似でキャッシュを効かせる。

`reuse_if_board_unchanged` (盤面一致による再利用、別テストファイル) とは
独立の別機構であり、本ファイルは量子化単体の効果を検証する。
"""
from __future__ import annotations

import inspect

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board
from src.board import HIDDEN_ROWS
import scripts.visualize_advantage_overlay as vao


def _board_with_ojama(n: int) -> Board:
    """可視領域下段から n 個のお邪魔を敷いた Board (他テストファイルの
    同名ヘルパーと同一設計、盤面bytesを変えるためだけに使う)。"""
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
    """`_reach` を軽量スタブに差し替え、呼出回数を計数する。"""
    calls = {"n": 0}

    def fake_reach(self, board, budget_sec, known_pairs,
                   threshold_ojama: float = vao.COUNTER_THRESHOLD_OJAMA) -> tuple[float, float]:
        calls["n"] += 1
        return (float(calls["n"]), float(calls["n"]) * 10.0)

    monkeypatch.setattr(vao.CounterReachTracker, "_reach", fake_reach)
    return calls


# ============================
# 量子化幅そのものが物理実測値由来であること (再フィット禁止の確認)
# ============================


def test_counter_budget_quantum_equals_beam_rollout_avg_step_time() -> None:
    """量子化幅は mc_counter_estimator.BEAM_ROLLOUT_AVG_STEP_TIME_SEC
    (PLACEMENT_SPEED_BY_ROW_SEC の単純平均) そのものであり、本ファイルで
    独自の再フィットは行っていない。"""
    assert vao.COUNTER_BUDGET_QUANTUM_SEC == pytest.approx(
        vao.mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC)
    # 実測値は0.13〜0.5秒レンジの平均であり、fable設計案の見立て(約0.35秒)
    # と整合する範囲に収まる (自己検収、マジックナンバーではないことの確認)。
    assert 0.30 < vao.COUNTER_BUDGET_QUANTUM_SEC < 0.40


# ============================
# CounterReachTracker._update_defender_only のロジック
# ============================


def test_quantize_off_by_default_uses_raw_precision_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既定 (quantize_budget_sec 省略=False) は従来通り `.2f` 精度で
    キャッシュキーを作る (backwards compat)。近接した budget_sec は
    別キーになり毎回再計算する。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    tracker.update(board, board, 5.00, defender_side="1P", threshold_ojama=6.0)
    tracker.update(board, board, 4.95, defender_side="1P", threshold_ojama=6.0)
    assert calls["n"] == 2


def test_quantize_on_merges_budget_values_in_same_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """quantize_budget_sec=True で、同一量子化バケットに入る budget_sec
    (打てる手数が同じはずの近い値) は2回目のキャッシュヒットとなり
    再計算しない。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    quantum = vao.COUNTER_BUDGET_QUANTUM_SEC
    bucket = 14
    b_lo = bucket * quantum + 0.01
    b_hi = bucket * quantum + quantum - 0.01
    assert int(b_lo // quantum) == int(b_hi // quantum) == bucket
    tracker.update(board, board, b_lo, defender_side="1P", threshold_ojama=6.0,
                    quantize_budget_sec=True)
    tracker.update(board, board, b_hi, defender_side="1P", threshold_ojama=6.0,
                    quantize_budget_sec=True)
    assert calls["n"] == 1  # 同一バケット -> 2回目はキャッシュヒット


def test_quantize_on_still_recomputes_across_bucket_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """量子化バケットが異なれば (=打てる手数が変わり得る) 再計算する。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    quantum = vao.COUNTER_BUDGET_QUANTUM_SEC
    tracker.update(board, board, quantum * 1.0, defender_side="1P", threshold_ojama=6.0,
                    quantize_budget_sec=True)
    tracker.update(board, board, quantum * 5.0, defender_side="1P", threshold_ojama=6.0,
                    quantize_budget_sec=True)
    assert calls["n"] == 2


def test_quantize_uses_original_budget_sec_for_actual_mc_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """キャッシュミス時に `_reach` へ渡される budget_sec は量子化前の元の値
    (量子化はキャッシュキーのみに影響し、実際の計算入力は変えない)。"""
    seen: list[float] = []

    def fake_reach(self, board, budget_sec, known_pairs,
                   threshold_ojama: float = vao.COUNTER_THRESHOLD_OJAMA) -> tuple[float, float]:
        seen.append(budget_sec)
        return (0.5, 3.0)

    monkeypatch.setattr(vao.CounterReachTracker, "_reach", fake_reach)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    exact_budget = 4.9123
    tracker.update(board, board, exact_budget, defender_side="1P", threshold_ojama=6.0,
                    quantize_budget_sec=True)
    assert seen == [exact_budget]  # 量子化されていない元の値がそのまま渡る


def test_quantize_and_placement_reuse_are_independent_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """量子化 (quantize_budget_sec) と盤面一致再利用 (reuse_if_board_unchanged)
    は独立に指定できる。量子化のみ ON・再利用 OFF でも、同一バケット内の
    budget_sec 変化はキャッシュヒットする (再利用フラグに依存しない)。"""
    calls = _patch_reach_with_counter(monkeypatch)
    tracker = vao.CounterReachTracker()
    board = _board_with_ojama(0)
    quantum = vao.COUNTER_BUDGET_QUANTUM_SEC
    bucket = 3
    tracker.update(board, board, bucket * quantum + 0.01, defender_side="1P",
                    threshold_ojama=6.0, quantize_budget_sec=True,
                    reuse_if_board_unchanged=False)
    tracker.update(board, board, bucket * quantum + quantum - 0.01, defender_side="1P",
                    threshold_ojama=6.0, quantize_budget_sec=True,
                    reuse_if_board_unchanged=False)
    assert calls["n"] == 1
    assert tracker._placement_reuse == {}  # 再利用側の状態には一切触れていない


# ============================
# 配線の署名・既定値・静的回帰テスト
# ============================


def test_counter_reach_tracker_update_signature_quantize_default_false() -> None:
    sig = inspect.signature(vao.CounterReachTracker.update)
    assert sig.parameters["quantize_budget_sec"].default is False


def test_resolved_exchange_tracker_budget_quantize_default_false() -> None:
    sig = inspect.signature(vao.ResolvedExchangeTracker.__init__)
    assert sig.parameters["enable_counter_budget_quantize"].default is False
    tracker = vao.ResolvedExchangeTracker(model=object())
    assert tracker._enable_counter_budget_quantize is False


def test_generate_signature_has_enable_resolved_counter_budget_quantize_default_false() -> None:
    sig = inspect.signature(vao.generate)
    assert "enable_resolved_counter_budget_quantize" in sig.parameters
    assert sig.parameters["enable_resolved_counter_budget_quantize"].default is False


def test_cli_resolved_counter_budget_quantize_flag_defaults_to_false() -> None:
    src = inspect.getsource(vao.main)
    assert '"--resolved-counter-budget-quantize"' in src
    assert 'dest="enable_resolved_counter_budget_quantize"' in src


def test_generate_source_wires_budget_quantize_to_both_constructions() -> None:
    """静的回帰テスト: generate() ソース中の ResolvedExchangeTracker 構築
    (通常時/試合境界リセット時の2箇所) が両方とも
    enable_counter_budget_quantize=enable_resolved_counter_budget_quantize を
    渡していることを固定する (配線漏れ防止)。"""
    src = inspect.getsource(vao.generate)
    code_only = src.replace(vao.generate.__doc__ or "", "")
    pattern = "enable_counter_budget_quantize=enable_resolved_counter_budget_quantize"
    assert code_only.count(pattern) == 2


def test_resolved_exchange_tracker_source_wires_quantize_flag_to_both_counter_calls() -> None:
    """静的回帰テスト: ResolvedExchangeTracker 内の counter_tracker.update 呼出
    2箇所 (_amplify_decisive / _reevaluate_live_defender) が両方とも
    quantize_budget_sec=self._enable_counter_budget_quantize を渡す。"""
    src = inspect.getsource(vao.ResolvedExchangeTracker)
    assert src.count("quantize_budget_sec=self._enable_counter_budget_quantize") == 2
