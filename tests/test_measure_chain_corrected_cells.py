"""tests/test_measure_chain_corrected_cells.py

連鎖補正版セル数評価の単体テスト。

テスト対象:
  - SideTracker.update(): ツモ / 連鎖 / おじゃま 遷移別の expected 更新ロジック
  - _compute_transitions_stats(): 統計集計
  - evaluate_match_chain_corrected(): 試合評価 dict の構造
  - _compute_overall_summary(): 全体統計集計
  - CLI --help / 引数なし
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.measure_chain_corrected_cells import (
    FAIL_ABS_ERR_MEAN_THRESHOLD,
    MIN_STABLE_WARMUP_COUNT,
    TSUMO_CELL_ADD,
    TSUMO_ERR_THRESHOLD,
    SideTracker,
    StableTransition,
    _compute_overall_summary,
    _compute_transitions_stats,
    evaluate_match_chain_corrected,
)
from src.board import Board, COLOR_RED, COLOR_EMPTY
from src.board_state_machine import BoardState


# ============================
# フィクスチャ
# ============================


def _make_board_with_cells(n_cells: int) -> Board:
    """非 EMPTY セルが n_cells 個の Board を返す。

    左下から順に COLOR_RED で埋める (row=12 から)。
    """
    b = Board()
    filled = 0
    for r in range(11, -1, -1):
        for c in range(6):
            if filled >= n_cells:
                return b
            b.set(r, c, COLOR_RED)
            filled += 1
    return b


def _make_transition(
    prev_state: str,
    recognized: int,
    expected: float,
    abs_err: float,
    is_tsumo: bool = False,
    is_chain: bool = False,
    is_ojama: bool = False,
) -> StableTransition:
    return StableTransition(
        time_sec=0.0,
        prev_state=prev_state,
        recognized=recognized,
        expected=expected,
        abs_err=abs_err,
        is_tsumo_transition=is_tsumo,
        is_chain_transition=is_chain,
        is_ojama_transition=is_ojama,
    )


# ============================
# SideTracker のウォームアップ処理
# ============================


def test_side_tracker_warmup_skips_transitions():
    """MIN_STABLE_WARMUP_COUNT 未満の STABLE では transitions に記録しないこと。"""
    tracker = SideTracker(side="1P")
    board = _make_board_with_cells(4)

    # ウォームアップ前の STABLE は記録されない
    for _ in range(MIN_STABLE_WARMUP_COUNT - 1):
        tracker.update(BoardState.STABLE, board, 1.0)

    assert len(tracker.transitions) == 0


def test_side_tracker_warmup_starts_recording_at_threshold():
    """MIN_STABLE_WARMUP_COUNT 回目の STABLE から記録が始まること。"""
    tracker = SideTracker(side="1P")
    board = _make_board_with_cells(4)

    for i in range(MIN_STABLE_WARMUP_COUNT):
        tracker.update(BoardState.STABLE, board, float(i))

    assert len(tracker.transitions) == 1


# ============================
# SideTracker: ツモ後 STABLE
# ============================


def _warmup_tracker(n_cells: int = 0) -> SideTracker:
    """ウォームアップ完了済みの SideTracker を返す。"""
    tracker = SideTracker(side="1P")
    board = _make_board_with_cells(n_cells)
    # ウォームアップ: MIN_STABLE_WARMUP_COUNT 回 STABLE を通す
    for i in range(MIN_STABLE_WARMUP_COUNT):
        tracker.update(BoardState.STABLE, board, float(i))
    return tracker


def test_tsumo_transition_adds_two_cells():
    """ツモ後 STABLE では expected が +2 されること。"""
    tracker = _warmup_tracker(n_cells=4)
    expected_before = tracker.expected

    # TSUMO_FALL → STABLE 遷移を模擬
    tracker.update(BoardState.TSUMO_FALL, None, 10.0)
    board_after = _make_board_with_cells(6)
    tracker.update(BoardState.STABLE, board_after, 11.0)

    last = tracker.transitions[-1]
    assert last.is_tsumo_transition
    assert abs(last.expected - (expected_before + TSUMO_CELL_ADD)) < 1e-9


def test_tsumo_transition_err_when_recognized_differs():
    """ツモ後 STABLE で recognized が expected と異なる場合 abs_err > 0 になること。"""
    tracker = _warmup_tracker(n_cells=4)
    # expected は 4 + 2 = 6 になるはずだが、認識は 5 しかない (誤認)
    tracker.update(BoardState.TSUMO_FALL, None, 10.0)
    board_bad = _make_board_with_cells(5)
    tracker.update(BoardState.STABLE, board_bad, 11.0)

    last = tracker.transitions[-1]
    assert last.abs_err > 0


def test_tsumo_transition_no_err_when_recognized_correct():
    """ツモ後 STABLE で recognized == expected なら abs_err == 0 になること。"""
    # ウォームアップで expected を known な値に合わせる
    tracker = SideTracker(side="1P")
    board_init = _make_board_with_cells(4)
    for i in range(MIN_STABLE_WARMUP_COUNT):
        tracker.update(BoardState.STABLE, board_init, float(i))

    # ウォームアップ直後の expected を確認
    expected_after_warmup = tracker.expected  # STABLE 連続なので +2 * warmup_count

    # TSUMO → STABLE: recognized = expected + 2
    tracker.update(BoardState.TSUMO_FALL, None, 10.0)
    new_recognized = int(expected_after_warmup) + TSUMO_CELL_ADD
    board_ok = _make_board_with_cells(new_recognized)
    tracker.update(BoardState.STABLE, board_ok, 11.0)

    last = tracker.transitions[-1]
    assert last.abs_err == 0.0


# ============================
# SideTracker: 連鎖後 STABLE
# ============================


def test_chain_transition_syncs_expected_to_recognized():
    """連鎖後 STABLE では expected が recognized に同期されること。"""
    tracker = _warmup_tracker(n_cells=20)

    # CHAIN → STABLE: 連鎖で 8 cell 消えて 12 cell になった
    tracker.update(BoardState.CHAIN, None, 20.0)
    board_after_chain = _make_board_with_cells(12)
    tracker.update(BoardState.STABLE, board_after_chain, 21.0)

    last = tracker.transitions[-1]
    assert last.is_chain_transition
    # 連鎖後は expected = recognized に同期するので abs_err = 0
    assert last.abs_err == 0.0
    assert abs(last.expected - 12) < 1e-9


# ============================
# SideTracker: おじゃま後 STABLE
# ============================


def test_ojama_transition_syncs_expected_to_recognized():
    """おじゃま後 STABLE では expected が recognized に同期されること。"""
    tracker = _warmup_tracker(n_cells=10)

    tracker.update(BoardState.OJAMA_FALL, None, 30.0)
    board_after_ojama = _make_board_with_cells(16)
    tracker.update(BoardState.STABLE, board_after_ojama, 31.0)

    last = tracker.transitions[-1]
    assert last.is_ojama_transition
    assert last.abs_err == 0.0
    assert abs(last.expected - 16) < 1e-9


# ============================
# SideTracker: STABLE 連続
# ============================


def test_stable_to_stable_adds_two():
    """STABLE → STABLE で recognized cell 数が増加した場合 expected が +2 されること。

    ウォームアップ完了後に recognized が 4 → 6 に変化した場合を検証する。
    STABLE 継続中でも recognized が変化すれば新 transition として記録される。
    """
    tracker = _warmup_tracker(n_cells=4)
    expected_before = tracker.expected
    n_transitions_before = len(tracker.transitions)

    # STABLE → STABLE: 同一盤面が確定したまま。まず別 cell 数の盤面で更新
    # (recognized=4 から recognized=6 に変化 → 着地直後とみなす)
    board_plus2 = _make_board_with_cells(6)
    tracker.update(BoardState.STABLE, board_plus2, 5.0)

    # 新しい transition が記録されていること
    assert len(tracker.transitions) == n_transitions_before + 1

    last = tracker.transitions[-1]
    assert not last.is_tsumo_transition
    assert not last.is_chain_transition
    assert not last.is_ojama_transition
    # expected = expected_before + 2 (STABLE→STABLE で +2)
    assert abs(last.expected - (expected_before + TSUMO_CELL_ADD)) < 1e-9


def test_stable_continues_no_new_transition():
    """STABLE 継続中に recognized cell 数が変化しない場合は transition が追加されないこと。"""
    tracker = _warmup_tracker(n_cells=4)
    n_transitions_before = len(tracker.transitions)

    # STABLE 継続: 同じ 4 cells の盤面を再度観測
    board_same = _make_board_with_cells(4)
    tracker.update(BoardState.STABLE, board_same, 5.0)

    # transition が増えていないこと
    assert len(tracker.transitions) == n_transitions_before


# ============================
# _compute_transitions_stats
# ============================


def test_stats_empty_list():
    """空リストで全フィールドが 0 になること。"""
    stats = _compute_transitions_stats([])
    assert stats["transition_count"] == 0
    assert stats["abs_err_mean"] == 0.0
    assert stats["abs_err_max"] == 0.0


def test_stats_mean_and_max():
    """abs_err_mean / abs_err_max が正しく計算されること。"""
    trs = [
        _make_transition("tsumo_fall", 6, 6.0, 0.0, is_tsumo=True),
        _make_transition("tsumo_fall", 7, 8.0, 1.0, is_tsumo=True),
        _make_transition("chain",      12, 12.0, 0.0, is_chain=True),
    ]
    stats = _compute_transitions_stats(trs)
    # errs = [0.0, 1.0, 0.0] → mean=1/3, max=1.0
    assert abs(stats["abs_err_mean"] - round(1.0 / 3, 4)) < 1e-4
    assert stats["abs_err_max"] == 1.0


def test_stats_tsumo_err_count():
    """TSUMO_ERR_THRESHOLD を超えるツモ誤認だけカウントされること。"""
    trs = [
        _make_transition("tsumo_fall", 6, 6.0, 0.0, is_tsumo=True),
        _make_transition("tsumo_fall", 5, 8.0, 3.0, is_tsumo=True),  # err > threshold
        _make_transition("tsumo_fall", 7, 8.0, 1.0, is_tsumo=True),  # err = threshold
    ]
    stats = _compute_transitions_stats(trs)
    # err=3.0 のみ > TSUMO_ERR_THRESHOLD(=1)、err=1.0 は境界 = カウントしない
    assert stats["tsumo_err_count"] == 1


def test_stats_counts_per_event_type():
    """連鎖 / おじゃま の件数が正しくカウントされること。"""
    trs = [
        _make_transition("chain",       12, 12.0, 0.0, is_chain=True),
        _make_transition("chain",       10, 10.0, 0.0, is_chain=True),
        _make_transition("ojama_fall",  16, 16.0, 0.0, is_ojama=True),
    ]
    stats = _compute_transitions_stats(trs)
    assert stats["chain_transition_count"] == 2
    assert stats["ojama_transition_count"] == 1


# ============================
# evaluate_match_chain_corrected
# ============================


def _make_tracker_with_transitions(
    side: str,
    errs: list[float],
) -> SideTracker:
    """指定した abs_err を持つ transitions を持つ SideTracker を返す。"""
    tracker = SideTracker(side=side)
    for e in errs:
        tracker.transitions.append(
            StableTransition(
                time_sec=0.0,
                prev_state="stable",
                recognized=10,
                expected=10.0 + e,
                abs_err=e,
                is_tsumo_transition=False,
                is_chain_transition=False,
                is_ojama_transition=False,
            )
        )
    return tracker


def test_evaluate_match_pass():
    """abs_err_mean が閾値以下なら PASS になること。"""
    t1 = _make_tracker_with_transitions("1P", [0.0, 0.5, 1.0])
    t2 = _make_tracker_with_transitions("2P", [0.0, 0.0, 0.0])
    result = evaluate_match_chain_corrected(1, t1, t2, 0.0, 60.0)
    assert result["verdict"] == "PASS"


def test_evaluate_match_fail():
    """abs_err_mean が閾値超なら FAIL になること。"""
    # FAIL_ABS_ERR_MEAN_THRESHOLD = 3.0 超
    t1 = _make_tracker_with_transitions("1P", [10.0, 10.0, 10.0])
    t2 = _make_tracker_with_transitions("2P", [0.0])
    result = evaluate_match_chain_corrected(1, t1, t2, 0.0, 60.0)
    assert result["verdict"] == "FAIL"


def test_evaluate_match_structure():
    """evaluate_match_chain_corrected が必要フィールドを全て持つこと。"""
    t1 = _make_tracker_with_transitions("1P", [1.0])
    t2 = _make_tracker_with_transitions("2P", [1.0])
    result = evaluate_match_chain_corrected(1, t1, t2, 10.0, 70.0)

    assert result["match_idx"] == 1
    assert result["start_sec"] == 10.0
    assert result["end_sec"] == 70.0
    assert "combined_abs_err_mean" in result
    assert "verdict" in result
    assert "p1" in result
    assert "p2" in result


def test_evaluate_match_empty_trackers():
    """transitions が空の場合に verdict PASS になること (err=0)。"""
    t1 = SideTracker(side="1P")
    t2 = SideTracker(side="2P")
    result = evaluate_match_chain_corrected(1, t1, t2, 0.0, 30.0)
    assert result["combined_abs_err_mean"] == 0.0
    assert result["verdict"] == "PASS"


# ============================
# _compute_overall_summary
# ============================


def test_overall_summary_empty():
    """空リストで verdict N/A になること。"""
    summary = _compute_overall_summary([])
    assert summary["verdict"] == "N/A"
    assert summary["match_count"] == 0


def test_overall_summary_single_video():
    """1 動画の summary が正しく集計されること。"""
    results = [{
        "video": "v40_match01.mp4",
        "matches": [{
            "match_idx": 1,
            "combined_abs_err_mean": 1.5,
            "verdict": "PASS",
            "p1": {"transition_count": 50, "abs_err_mean": 1.5,
                   "abs_err_max": 5.0, "tsumo_transition_count": 40,
                   "tsumo_err_count": 2, "chain_transition_count": 5,
                   "ojama_transition_count": 5},
            "p2": {"transition_count": 50, "abs_err_mean": 1.5,
                   "abs_err_max": 4.0, "tsumo_transition_count": 40,
                   "tsumo_err_count": 3, "chain_transition_count": 5,
                   "ojama_transition_count": 5},
        }],
        "summary": {"combined_abs_err_mean": 1.5, "verdict": "PASS"},
    }]
    summary = _compute_overall_summary(results)
    assert summary["match_count"] == 1
    assert summary["fail_match_count"] == 0
    assert summary["verdict"] == "PASS"


def test_overall_summary_fail_propagates():
    """FAIL 試合があるとき fail_match_count が正しくカウントされること。"""
    results = [
        {
            "video": "va.mp4",
            "matches": [{
                "match_idx": 1,
                "combined_abs_err_mean": 5.0,
                "verdict": "FAIL",
                "p1": {"transition_count": 10, "abs_err_mean": 5.0,
                       "abs_err_max": 10.0, "tsumo_transition_count": 8,
                       "tsumo_err_count": 5, "chain_transition_count": 1,
                       "ojama_transition_count": 1},
                "p2": {"transition_count": 10, "abs_err_mean": 5.0,
                       "abs_err_max": 10.0, "tsumo_transition_count": 8,
                       "tsumo_err_count": 5, "chain_transition_count": 1,
                       "ojama_transition_count": 1},
            }],
        },
        {
            "video": "vb.mp4",
            "matches": [{
                "match_idx": 1,
                "combined_abs_err_mean": 1.0,
                "verdict": "PASS",
                "p1": {"transition_count": 50, "abs_err_mean": 1.0,
                       "abs_err_max": 3.0, "tsumo_transition_count": 45,
                       "tsumo_err_count": 2, "chain_transition_count": 3,
                       "ojama_transition_count": 2},
                "p2": {"transition_count": 50, "abs_err_mean": 1.0,
                       "abs_err_max": 3.0, "tsumo_transition_count": 45,
                       "tsumo_err_count": 2, "chain_transition_count": 3,
                       "ojama_transition_count": 2},
            }],
        },
    ]
    summary = _compute_overall_summary(results)
    assert summary["match_count"] == 2
    assert summary["fail_match_count"] == 1


# ============================
# CLI テスト
# ============================


def test_cli_no_args_exits_nonzero():
    """引数なしで実行すると非 0 で終了すること。"""
    ret = subprocess.run(
        [sys.executable, "scripts/measure_chain_corrected_cells.py"],
        capture_output=True,
        cwd=str(_PROJ_ROOT),
    )
    assert ret.returncode != 0


def test_cli_help_exits_zero():
    """--help で exit code 0 になること。"""
    ret = subprocess.run(
        [sys.executable, "scripts/measure_chain_corrected_cells.py", "--help"],
        capture_output=True,
        cwd=str(_PROJ_ROOT),
    )
    assert ret.returncode == 0
