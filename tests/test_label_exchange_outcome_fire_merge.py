"""発火イベント分裂統合 (測定器事故5件目、2026-08-02) の回帰テスト。

scripts/label_exchange_outcome.py の _merge_fire_event_clusters /
_last_valid_score_before / _process_game への配線を、軽量な合成データのみで
検証する (実動画・実npzは使わない、tests/test_exchange_effectiveness_step2.py
と同じ方針)。

検収基準:
  1. 分裂パターン (部分加算2回、盤面凍結 or 短ギャップ) が1イベントに統合される
  2. 正当な連続発火 (盤面変化あり・gap>2.5秒) は2イベントのまま分離される
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_RED
from src.chain import ChainSimulator
from scripts.label_exchange_outcome import (
    FIRE_EVENT_MERGE_GAP_SEC,
    NpzRecord,
    _last_valid_score_before,
    _merge_fire_event_clusters,
    _process_game,
)


# =============================================================================
# _last_valid_score_before
# =============================================================================

class TestLastValidScoreBefore:
    def test_finds_nearest_valid_score(self) -> None:
        score = np.array([100, -1, -1, 200, 300])
        assert _last_valid_score_before(score, index=3) == 100

    def test_returns_minus_one_when_none_before(self) -> None:
        score = np.array([-1, -1, 500])
        assert _last_valid_score_before(score, index=2) == -1

    def test_skips_multiple_missing_values(self) -> None:
        score = np.array([50, -1, -1, -1, -1, 999])
        assert _last_valid_score_before(score, index=5) == 50


# =============================================================================
# _merge_fire_event_clusters
# =============================================================================

def _empty_grids(n: int) -> np.ndarray:
    """全フレーム空盤面 (13,6) の合成グリッド配列。"""
    return np.zeros((n, BOARD_ROWS, BOARD_COLS), dtype=np.int8)


class TestMergeFireEventClusters:
    """測定器事故5件目: 分裂検出の統合ロジック。"""

    def test_merges_split_with_matching_grid_and_short_gap(self) -> None:
        """盤面凍結 (grid一致) + 短ギャップ = 実データ (c27 game12) と同型の分裂。"""
        t_sec = np.array([0.0, 10.0, 12.0])
        score = np.array([100, 27590, 35274])
        grids = _empty_grids(3)  # 全フレーム同一 (連鎖中の凍結を模擬)
        fire_indices = [1, 2]
        clusters = _merge_fire_event_clusters(t_sec, score, grids, fire_indices)
        assert len(clusters) == 1
        assert clusters[0].fire_index == 2  # 連鎖終了時点 (最後の検出)
        assert clusters[0].board_ref_index == 0  # クラスタ先頭(=1)の1つ前
        assert clusters[0].baseline_score == 100  # クラスタ先頭直前の有効スコア

    def test_merges_via_short_gap_even_when_grid_differs(self) -> None:
        """主信号 (grid一致) が抜けても副信号 (短ギャップ) がマージする保険。"""
        t_sec = np.array([0.0, 10.0, 11.5])
        score = np.array([100, 500, 900])
        grids = _empty_grids(3)
        grids[2, 0, 0] = COLOR_RED  # 最終検出だけ盤面が更新済み (grid不一致)
        fire_indices = [1, 2]
        clusters = _merge_fire_event_clusters(t_sec, score, grids, fire_indices)
        assert len(clusters) == 1
        assert clusters[0].fire_index == 2
        assert clusters[0].baseline_score == 100

    def test_keeps_separate_when_grid_differs_and_gap_large(self) -> None:
        """正当な連続発火 (盤面変化あり・gap>2.5秒) は分離を維持する。"""
        t_sec = np.array([0.0, 10.0, 20.0])
        score = np.array([100, 500, 1000])
        grids = _empty_grids(3)
        grids[2, 0, 0] = COLOR_RED  # 盤面が変わっている (別の連鎖)
        fire_indices = [1, 2]
        clusters = _merge_fire_event_clusters(t_sec, score, grids, fire_indices)
        assert len(clusters) == 2
        assert [c.fire_index for c in clusters] == [1, 2]
        assert clusters[1].baseline_score == 500  # 2件目は独立 (直前有効scoreそのまま)

    def test_gap_exactly_at_threshold_merges(self) -> None:
        """ギャップがちょうど閾値 (FIRE_EVENT_MERGE_GAP_SEC) ならマージする (境界値)。"""
        t_sec = np.array([0.0, 10.0, 10.0 + FIRE_EVENT_MERGE_GAP_SEC])
        score = np.array([100, 500, 900])
        grids = _empty_grids(3)
        grids[2, 0, 0] = COLOR_RED
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 2])
        assert len(clusters) == 1

    def test_three_way_split_merges_into_single_cluster(self) -> None:
        """3分割 (部分加算3回) も1イベントに統合される (連鎖単位クラスタリング)。"""
        t_sec = np.array([0.0, 10.0, 11.0, 12.0])
        score = np.array([100, 300, 600, 1000])
        grids = _empty_grids(4)  # 全フレーム同一グリッド (連鎖中の凍結)
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 2, 3])
        assert len(clusters) == 1
        assert clusters[0].fire_index == 3
        assert clusters[0].baseline_score == 100

    def test_empty_fire_indices_returns_empty_list(self) -> None:
        t_sec = np.array([0.0, 1.0])
        score = np.array([0, 100])
        grids = _empty_grids(2)
        assert _merge_fire_event_clusters(t_sec, score, grids, []) == []


# =============================================================================
# _process_game (統合テスト、検収基準3)
# =============================================================================

def _make_record(video_id: str, side: str, t_sec: list, score: list, grids: np.ndarray) -> NpzRecord:
    """テスト用 NpzRecord を組み立てる (won=常に0、game_idx=常に0)。"""
    n = len(t_sec)
    return NpzRecord(
        video_id=video_id, side=side,
        t_sec=np.array(t_sec, dtype=np.float32), game_idx=np.zeros(n, dtype=np.int32),
        grids=grids, won=np.zeros(n, dtype=np.float32), score=np.array(score, dtype=np.int32),
    )


class TestProcessGameFireMergeIntegration:
    """_process_game の出力件数レベルで統合/分離の両方向を確認する (検収基準3)。"""

    def test_split_pattern_yields_single_row(self) -> None:
        """分裂パターン (部分加算2回、盤面凍結) は1行に統合される

        (実データ c27.npz 1P game12 と同型: 27590→35274 の分裂が同一9連鎖)。
        """
        # idx0,1: 非発火の準備フレーム (delta<SCORE_DELTA_FIRE)。
        # idx2,3: 分裂した同一連鎖 (盤面凍結=grid一致、gap=2秒<=閾値)。
        t_sec = [0.0, 1.0, 10.0, 12.0]
        score = [0, 20, 27590, 35274]
        grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)  # 全フレーム同一盤面
        fire_rec = _make_record("video_test", "1P", t_sec, score, grids)
        opp_grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        opp_rec = _make_record("video_test", "2P", t_sec, [0, 0, 0, 0], opp_grids)

        sim = ChainSimulator()
        rows = _process_game(fire_rec, opp_rec, "1P", sim, puyo_q_low=10.0, puyo_q_high=30.0)
        assert len(rows) == 1
        # net_ojama は標準レート70点/個換算: (35274 - クラスタ先頭直前スコア20) // 70
        expected_ojama = (35274 - 20) // 70
        assert rows[0]["net_ojama"] == pytest.approx(float(expected_ojama))

    def test_legit_separate_fires_yield_two_rows(self) -> None:
        """盤面変化あり・gap>2.5秒の正当な連続発火は2行に分離される。"""
        t_sec = [0.0, 1.0, 10.0, 30.0]
        score = [0, 50, 500, 1200]
        grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        grids[3, 0, 0] = COLOR_RED  # 2件目の発火は盤面が変わっている
        fire_rec = _make_record("video_test", "1P", t_sec, score, grids)
        opp_grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        opp_rec = _make_record("video_test", "2P", t_sec, [0, 0, 0, 0], opp_grids)

        sim = ChainSimulator()
        rows = _process_game(fire_rec, opp_rec, "1P", sim, puyo_q_low=10.0, puyo_q_high=30.0)
        assert len(rows) == 2
