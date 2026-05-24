"""ChainPhaseDetector のテスト (Phase Z-1)。"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_BLUE, COLOR_EMPTY,
    COLOR_RED, HIDDEN_ROWS,
)
from src.chain_phase_detector import (
    ChainPhaseDetector, SCORE_DELTA_FIRE,
)


def _make_4_red_cluster_board() -> Board:
    """最下段 col 0..3 に赤 4 連結を置いた盤面 (発火可能)。"""
    board = Board()
    last_row = BOARD_ROWS - 1
    for c in range(4):
        board.set(last_row, c, COLOR_RED)
    return board


def _make_no_chain_board() -> Board:
    """発火不可な盤面 (赤 3 個だけ)。"""
    board = Board()
    last_row = BOARD_ROWS - 1
    for c in range(3):
        board.set(last_row, c, COLOR_RED)
    return board


def test_no_score_delta_no_chain() -> None:
    detector = ChainPhaseDetector()
    board = _make_4_red_cluster_board()
    res = detector.update(0.0, board, board, 1000, 1000)
    assert not res.is_chain_p1
    assert not res.is_chain_p2


def test_score_jump_with_erasable_triggers_chain() -> None:
    detector = ChainPhaseDetector()
    erasable = _make_4_red_cluster_board()
    # 1 frame 目: 安定 (連鎖前) score=1000
    detector.update(0.0, erasable, erasable, 1000, 1000)
    # 2 frame 目: score 急増、4+ cluster 存在 → 連鎖中フラグ on
    res = detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    assert res.is_chain_p1
    assert not res.is_chain_p2


def test_score_jump_without_erasable_still_triggers() -> None:
    """Z-3A: 4+ cluster が直前 frame になくても score 急増で連鎖中扱い。

    CellRecoveryRefiner で cluster が一時的に崩れているケースや、
    隠し段の落下による連鎖でも発火検出するための仕様。
    predicted_board は生成されないが is_chain=True で suspicious 判定が skip される。
    """
    detector = ChainPhaseDetector()
    no_chain = _make_no_chain_board()
    detector.update(0.0, no_chain, no_chain, 1000, 1000)
    res = detector.update(0.1, no_chain, no_chain, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    assert res.is_chain_p1
    # predicted_board は erasable がないので None
    assert res.predicted_p1 is None


def test_chain_finishes_after_score_still() -> None:
    """Z-3C: tail buffer を 0 にして純粋な完了判定をテスト。"""
    detector = ChainPhaseDetector(chain_tail_buffer_sec=0.0)
    erasable = _make_4_red_cluster_board()
    detector.update(0.0, erasable, erasable, 1000, 1000)
    # 発火 frame
    detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    # score=2000 に上昇して以降、3 frame の比較が連続で同値になるまで待つ
    detector.update(0.2, erasable, erasable, 2000, 1000)  # last_score 更新
    detector.update(0.3, erasable, erasable, 2000, 1000)  # still_count=1
    detector.update(0.4, erasable, erasable, 2000, 1000)  # still_count=2
    res = detector.update(0.5, erasable, erasable, 2000, 1000)  # still_count=3 → 完了
    assert not res.is_chain_p1


def test_tail_buffer_keeps_chain_active() -> None:
    """Z-3C: 完了後 tail_buffer_sec 以内は連鎖中扱い維持。"""
    detector = ChainPhaseDetector(chain_tail_buffer_sec=0.5)
    erasable = _make_4_red_cluster_board()
    detector.update(0.0, erasable, erasable, 1000, 1000)
    detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    # 静止 → 完了
    detector.update(0.2, erasable, erasable, 2000, 1000)
    detector.update(0.3, erasable, erasable, 2000, 1000)
    detector.update(0.4, erasable, erasable, 2000, 1000)
    detector.update(0.5, erasable, erasable, 2000, 1000)  # 完了 (still=3)
    # 完了後 0.3s: tail buffer 内で連鎖中維持
    res = detector.update(0.8, erasable, erasable, 2000, 1000)
    assert res.is_chain_p1
    # 完了後 0.6s: tail buffer 超過で off
    res2 = detector.update(1.1, erasable, erasable, 2000, 1000)
    assert not res2.is_chain_p1


def test_chain_timeout_finishes() -> None:
    detector = ChainPhaseDetector(chain_hold_sec=0.5)
    erasable = _make_4_red_cluster_board()
    detector.update(0.0, erasable, erasable, 1000, 1000)
    detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    # timeout 経過で完了 (score 変化があってもよい)
    res = detector.update(0.7, erasable, erasable, 5000, 1000)
    assert not res.is_chain_p1


def test_predicted_board_provided_during_chain() -> None:
    detector = ChainPhaseDetector()
    erasable = _make_4_red_cluster_board()
    detector.update(0.0, erasable, erasable, 1000, 1000)
    res = detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    assert res.is_chain_p1
    assert res.predicted_p1 is not None
    # 4 つ消えて empty になっているはず
    last_row = BOARD_ROWS - 1
    for c in range(4):
        assert res.predicted_p1.get(last_row, c) == COLOR_EMPTY


def test_board_diff_fires_chain_without_score() -> None:
    """Z-3B: score 欠損中でも puyo cell 数が急減したら連鎖中扱い。"""
    detector = ChainPhaseDetector()
    erasable = _make_4_red_cluster_board()
    # 初期盤面: 4 puyo
    detector.update(0.0, erasable, erasable, None, None)
    # 次 frame: 全 puyo 消滅 (puyo_diff=4)
    empty = Board()
    res = detector.update(0.1, empty, empty, None, None)
    assert res.is_chain_p1
    assert res.is_chain_p2


def test_board_diff_does_not_fire_below_threshold() -> None:
    """Z-3B: puyo cell 数の減少が 4 未満なら発火しない。"""
    detector = ChainPhaseDetector(puyo_diff_fire=4)
    # 3 個減少: 連鎖発火しない
    b1 = Board()
    last_row = BOARD_ROWS - 1
    for c in range(4):
        b1.set(last_row, c, COLOR_RED)
    b2 = Board()
    b2.set(last_row, 0, COLOR_RED)  # 1 個残し (3 個減少)
    detector.update(0.0, b1, b1, None, None)
    res = detector.update(0.1, b2, b2, None, None)
    assert not res.is_chain_p1


def test_reset_clears_state() -> None:
    detector = ChainPhaseDetector()
    erasable = _make_4_red_cluster_board()
    detector.update(0.0, erasable, erasable, 1000, 1000)
    detector.update(0.1, erasable, erasable, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    detector.reset()
    res = detector.update(0.0, erasable, erasable, 1000, 1000)
    assert not res.is_chain_p1


# === cycle 71d (案 D6): puyo_diff 偽発火の noise recovery 取り消しテスト ===


def _make_n_red_cluster_board(n: int) -> Board:
    """col=0 に縦積み n 個 (= BOARD_COLS=6 制約に依存しない)."""
    board = Board()
    last_row = BOARD_ROWS - 1
    for i in range(n):
        board.set(last_row - i, 0, COLOR_RED)
    return board


def test_d6_diff_fire_canceled_when_puyo_count_recovers() -> None:
    """案 D6: puyo_diff のみで発火後、 短時間で baseline に戻ったら取り消し.

    cycle 71c β 系の cnn=32→27→32 振動を再現. 1 frame だけ -5 cells スパイクで
    PUYO_DIFF_FIRE 発火するが、 次 frame で puyo_count が戻ったら chain 取り消し.
    """
    detector = ChainPhaseDetector(puyo_diff_fire=4)
    full = _make_n_red_cluster_board(8)  # 8 cells
    drop = _make_n_red_cluster_board(3)  # -5 cells スパイク
    # 初期 stable (= 8 cells)
    detector.update(0.0, full, full, 1000, 1000)
    # 単発スパイク (= 8 → 3) で diff_fire 発火 (= score 増加なし)
    res1 = detector.update(0.1, drop, drop, 1000, 1000)
    assert res1.is_chain_p1
    # 0.05s 後に puyo_count が baseline (= 8) に戻る → noise 判定で chain 取り消し
    res2 = detector.update(0.15, full, full, 1000, 1000)
    assert not res2.is_chain_p1


def test_d6_real_chain_not_canceled() -> None:
    """案 D6: score 増加 (= 真の連鎖) は noise recovery 取り消し対象外."""
    detector = ChainPhaseDetector()
    full = _make_n_red_cluster_board(8)
    drop = _make_n_red_cluster_board(3)
    detector.update(0.0, full, full, 1000, 1000)
    # score 増加 + diff 両方発火
    res1 = detector.update(0.1, drop, drop, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    assert res1.is_chain_p1
    # puyo_count 戻っても score_fire 経由なので取り消されない
    res2 = detector.update(0.15, full, full, 1000 + SCORE_DELTA_FIRE * 2, 1000)
    assert res2.is_chain_p1


def test_d6_recovery_after_window_does_not_cancel() -> None:
    """案 D6: NOISE_RECOVERY_WINDOW_SEC を超えてから戻った場合は取り消ししない."""
    detector = ChainPhaseDetector()
    full = _make_n_red_cluster_board(8)
    drop = _make_n_red_cluster_board(3)
    detector.update(0.0, full, full, None, None)
    # diff_fire 発火 (= score None で diff only)
    res1 = detector.update(0.1, drop, drop, None, None)
    assert res1.is_chain_p1
    # 0.3s 後 (= 200ms window 超え) に戻っても取り消ししない (= 連鎖継続)
    res2 = detector.update(0.4, full, full, None, None)
    assert res2.is_chain_p1


def test_d6_continued_chain_not_canceled() -> None:
    """案 D6: puyo cells が継続的に減少する真の連鎖は取り消さない."""
    detector = ChainPhaseDetector()
    full = _make_n_red_cluster_board(8)
    drop1 = _make_n_red_cluster_board(3)  # -5
    drop2 = _make_n_red_cluster_board(0)  # 全消し
    detector.update(0.0, full, full, None, None)
    res1 = detector.update(0.1, drop1, drop1, None, None)
    assert res1.is_chain_p1
    # baseline まで戻らず、 さらに減少 → 連鎖継続
    res2 = detector.update(0.15, drop2, drop2, None, None)
    assert res2.is_chain_p1
