"""反復9 ガード3 (2026-07-23): 物理レビュー純粋関数の陽性/陰性ユニットテスト。

scripts/recognition_physics_review.py の `_measure_ghost_mismatch` は
「連鎖後最初の STABLE confirmed_board が ChainSimulator.simulate(連鎖前盤面)
.final_board と一致するか」を測る中核メトリクスであり、反復2(根治)/
反復5-6(物理推論スルー・答え合わせ)の効果測定に直接使われた。

本テストは固定盤面 (動画デコード不要、軽量) で:
  - 陽性ケース: 連鎖後 STABLE が物理予測と一致 → mismatch_cells == 0
  - 陰性ケース (残像バグ相当): 連鎖後 STABLE が連鎖前の色のまま残る
    (= 反復1で修正した「残像」症状そのもの) → mismatch_cells > 0
を検知できることを確認し、このメトリクス自体が壊れて「常に緑」になる
退行を防ぐ。

重い実動画版 (scripts/recognition_physics_review.py の _capture_frames 等を
使うフル E2E テスト) は別途 @pytest.mark.slow でマークし、既定の
`pytest tests/` では実行しない (nightly 枠向け、pytest.ini の
`addopts = -m "not slow"` により既定 skip)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import COLOR_EMPTY, COLOR_GREEN, COLOR_RED, Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from scripts.recognition_physics_review import (  # noqa: E402
    _FrameRecord, _measure_ghost_mismatch,
)


def _erasable_board_with_survivor() -> Board:
    """4連結の赤 (row12 col1-4) + その上の緑 (row11 col0) の盤面。

    ChainSimulator.simulate() で chain_count=1、final_board は
    (12,0)=緑 (重力落下)、それ以外は空になる
    (tests/test_recognition_pipeline.py の同名ヘルパーと同一仕様)。
    """
    b = Board()
    b.set(12, 1, COLOR_RED)
    b.set(12, 2, COLOR_RED)
    b.set(12, 3, COLOR_RED)
    b.set(12, 4, COLOR_RED)
    b.set(11, 0, COLOR_GREEN)
    return b


def _make_trigger_and_stable_records(
    stable_grid_board: Board,
) -> list[_FrameRecord]:
    """1 連鎖トリガー frame + 直後の STABLE frame からなる records を作る。

    Args:
        stable_grid_board: 連鎖後最初の STABLE で観測された盤面
            (陽性ケースなら物理予測と同じ盤面、陰性ケースなら連鎖前のまま)。
    """
    before = _erasable_board_with_survivor()
    return [
        _FrameRecord(
            frame_idx=0, t_sec=1.0, state="CHAIN", grid=None, score=0,
            chain_trigger_sec=1.0, chain_before_grid=before._grid.copy(),
            chain_ojama_sent=0, n_erasure_alerts=0, n_transition_drop_alerts=0,
        ),
        _FrameRecord(
            frame_idx=1, t_sec=1.5, state="STABLE",
            grid=stable_grid_board._grid.copy(), score=0,
            chain_trigger_sec=None, chain_before_grid=None,
            chain_ojama_sent=None, n_erasure_alerts=0,
            n_transition_drop_alerts=0,
        ),
    ]


def test_measure_ghost_mismatch_positive_case_zero_when_physics_matches() -> None:
    """陽性ケース: STABLE 盤面が物理予測 (final_board) と一致すれば
    mismatch_cells == 0 を検知する (誤検知しないことの確認)。
    """
    correct_final = Board()
    correct_final.set(12, 0, COLOR_GREEN)
    records = _make_trigger_and_stable_records(correct_final)
    sim = ChainSimulator()
    results = _measure_ghost_mismatch(records, sim)
    assert len(results) == 1
    assert results[0]["mismatch_cells"] == 0
    assert results[0]["resolved"] is True


def test_measure_ghost_mismatch_negative_case_detects_ghost_residue() -> None:
    """陰性ケース (残像バグ相当): STABLE 盤面が連鎖前の色のまま残っている
    (= 反復1で修正した症状そのもの) 場合、mismatch_cells > 0 を検知する。
    """
    stale_board = _erasable_board_with_survivor()  # 連鎖前のまま (赤+緑残留)
    records = _make_trigger_and_stable_records(stale_board)
    sim = ChainSimulator()
    results = _measure_ghost_mismatch(records, sim)
    assert len(results) == 1
    assert results[0]["mismatch_cells"] > 0, (
        "残像 (連鎖前の色が STABLE に残る) を mismatch として検知すべき"
    )


def test_measure_ghost_mismatch_ignores_no_real_chain() -> None:
    """起点盤面に連鎖可能なグループが無い (疑似イベント) 場合は対象外にする。"""
    empty_before = Board()
    records = [
        _FrameRecord(
            frame_idx=0, t_sec=1.0, state="CHAIN", grid=None, score=0,
            chain_trigger_sec=1.0, chain_before_grid=empty_before._grid.copy(),
            chain_ojama_sent=0, n_erasure_alerts=0, n_transition_drop_alerts=0,
        ),
        _FrameRecord(
            frame_idx=1, t_sec=1.5, state="STABLE", grid=empty_before._grid.copy(),
            score=0, chain_trigger_sec=None, chain_before_grid=None,
            chain_ojama_sent=None, n_erasure_alerts=0,
            n_transition_drop_alerts=0,
        ),
    ]
    sim = ChainSimulator()
    results = _measure_ghost_mismatch(records, sim)
    assert results == []


# ============================
# 重い実動画版 (nightly 枠、既定 skip)
# ============================


@pytest.mark.slow
def test_measure_ghost_mismatch_real_video_smoke() -> None:
    """重い実動画版スモークテスト (nightly 枠)。

    scripts/recognition_physics_review.py の _capture_frames を使い、
    実際の動画 1本分を処理して例外なく完走することを確認する
    (ここでは数値の当否までは問わない、E2E 経路の生存確認)。
    既定の `pytest tests/` では pytest.ini の `addopts = -m "not slow"`
    により自動的に除外される。
    """
    from scripts.recognition_physics_review import _capture_frames, TARGET_WINDOWS
    stem, start_sec, max_sec = TARGET_WINDOWS[0]
    by_side = _capture_frames(stem, start_sec, min(max_sec, 10.0))
    assert "1P" in by_side and "2P" in by_side
