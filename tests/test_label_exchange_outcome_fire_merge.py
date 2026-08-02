"""発火イベント分裂統合 (測定器事故5件目、2026-08-02 v3) の回帰テスト。

scripts/label_exchange_outcome.py の _merge_fire_event_clusters /
_has_placement_signature / _last_valid_score_before / _process_game への
配線を、軽量な合成データのみで検証する (実動画・実npzは使わない、
tests/test_exchange_effectiveness_step2.py と同じ方針)。

v3 (2026-08-02、main実測診断で確定):
  - v2の主判定「候補検出j自身の盤面が参照と一致するか」は、連鎖終了→盤面が
    連鎖後STABLEに更新→最終スコア確定(=検出j)という正当な順序でも
    誤って「不一致」と判定してしまい、gap1.5-2.2秒帯域152件中148件を
    誤分離していた (main実測)。
  - v3は判定基準を「ぷよ総数の持続的な+2以上増加 (=設置の署名)」の有無に
    全面変更する。増加が無ければ (凍結のまま or 連鎖後の減少のみ) マージする。
  - 持続判定秒数は「認識ノイズの自己修復秒数(≤1秒)」ではなく「1手の所要時間
    (SEC_PER_HAND)」を使う (自己テストで発見した回帰: 前者を使うと、設置
    直後に連鎖が発火してぷよが消えるケースを誤ってノイズ扱いしてしまう)。

検収基準:
  1. 分裂パターン (部分加算2回、ぷよ数増加なし) が1イベントに統合される
  2. 正当な連続発火 (設置の署名あり) は2イベントのまま分離される
  3. (a) 連鎖後更新フレームを挟む尻尾分裂→マージ (v2規則では落ちるテスト)
     (b) 設置署名 (+2増加が持続) を挟む gap2.2秒→分離維持
     (c) +1ノイズのみ→マージ
"""
from __future__ import annotations

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_RED, COLOR_BLUE, COLOR_UNKNOWN
from src.chain import ChainSimulator
from scripts.label_exchange_outcome import (
    FIRE_EVENT_MERGE_GAP_SEC,
    PLACEMENT_SIGNATURE_MIN_INCREASE,
    PLACEMENT_SIGNATURE_PERSIST_SEC,
    NpzRecord,
    _count_concrete_puyos,
    _has_placement_signature,
    _last_valid_score_before,
    _merge_fire_event_clusters,
    _process_game,
)


# =============================================================================
# _last_valid_score_before (v1から不変)
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
# _count_concrete_puyos
# =============================================================================

def _empty_grids(n: int) -> np.ndarray:
    """全フレーム空盤面 (13,6) の合成グリッド配列。"""
    return np.zeros((n, BOARD_ROWS, BOARD_COLS), dtype=np.int8)


class TestCountConcretePuyos:
    def test_counts_colored_cells(self) -> None:
        grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        grid[0, 0] = COLOR_RED
        grid[0, 1] = COLOR_BLUE
        assert _count_concrete_puyos(grid) == 2

    def test_excludes_unknown_cells(self) -> None:
        """遷移汚染で湧くUNKNOWN(10)は非ゼロだがカウント対象外

        (Board.height_of()と同じ方針、main指摘事項)。
        """
        grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        grid[0, 0] = COLOR_RED
        grid[1, 0] = COLOR_UNKNOWN
        assert _count_concrete_puyos(grid) == 1

    def test_empty_grid_is_zero(self) -> None:
        assert _count_concrete_puyos(np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)) == 0


# =============================================================================
# _has_placement_signature (v3主判定)
# =============================================================================

class TestHasPlacementSignature:
    def test_no_change_returns_false(self) -> None:
        t_sec = np.array([0.0, 1.0, 2.0])
        grids = _empty_grids(3)
        assert _has_placement_signature(t_sec, grids, 0, 2) is False

    def test_decrease_only_returns_false(self) -> None:
        """連鎖後の更新 (ぷよが減る) だけでは設置の署名にならない (v2からの主な変更点)。"""
        grids = _empty_grids(3)
        grids[0, 0, 0] = COLOR_RED
        grids[0, 0, 1] = COLOR_BLUE
        grids[0, 0, 2] = COLOR_RED  # 開始時点は3個、以降は減るだけ
        t_sec = np.array([0.0, 1.0, 2.0])
        assert _has_placement_signature(t_sec, grids, 0, 2) is False

    def test_transient_plus1_noise_returns_false(self) -> None:
        """+1個だけの写り込みは閾値未満のため署名にならない (検収基準3c)。"""
        t_sec = np.array([0.0, 1.0, 2.0])
        grids = _empty_grids(3)
        grids[1, 0, 0] = COLOR_RED  # 中間フレームだけ+1個
        assert _has_placement_signature(t_sec, grids, 0, 2) is False

    def test_transient_plus2_that_reverts_quickly_returns_false(self) -> None:
        """+2以上でも SEC_PER_HAND 未満で解消するなら認識ノイズとみなし署名にしない

        (実データ c27 game12 1P の0.2秒ノイズを模擬)。
        """
        t_sec = np.array([0.0, 0.1, 0.2, 0.3])
        grids = _empty_grids(4)
        grids[1, 0, 0] = COLOR_RED
        grids[1, 0, 1] = COLOR_BLUE  # t=0.1: +2個
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE  # t=0.2: 継続(経過0.1秒、閾値未満)
        # t=0.3 (index3, 候補j自身): 参照に復帰 (0個)
        assert _has_placement_signature(t_sec, grids, 0, 3) is False

    def test_sustained_increase_beyond_persist_sec_returns_true(self) -> None:
        """+2以上の増加が PLACEMENT_SIGNATURE_PERSIST_SEC を超えて持続すれば

        設置ありと確定する (その後連鎖で減っても遅すぎるので確定は覆らない、
        実データ c27 game12 1P の index47→52 と同型の回帰ガード)。
        """
        persist = PLACEMENT_SIGNATURE_PERSIST_SEC
        # k=1で+2個のジャンプ開始 (t=0.1)、k=2の時点で経過persist*1.1秒(>persist)。
        t_sec = np.array([0.0, 0.1, 0.1 + persist * 1.1, 0.1 + persist * 1.2])
        grids = _empty_grids(4)
        grids[1, 0, 0] = COLOR_RED
        grids[1, 0, 1] = COLOR_BLUE  # +2個 (ジャンプ開始)
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE  # 継続 (経過persist*1.1秒 > persist で確定)
        # index3 (候補j自身) は連鎖で消えて0個に戻ってもよい (既に確定済みのため無関係)
        assert _has_placement_signature(t_sec, grids, 0, 3) is True


# =============================================================================
# _merge_fire_event_clusters (v3)
# =============================================================================

class TestMergeFireEventClusters:
    """測定器事故5件目 v3: 分裂検出の統合ロジック。"""

    def test_merges_split_with_no_placement_signature(self) -> None:
        """分裂パターン (部分加算2回、ぷよ数の増加なし) は1イベントに統合される

        (検収基準1と同型)。
        """
        t_sec = np.array([0.0, 10.0, 12.0])
        score = np.array([100, 27590, 35274])
        grids = _empty_grids(3)  # 増加なし (凍結)
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 2])
        assert len(clusters) == 1
        assert clusters[0].fire_index == 2
        assert clusters[0].board_ref_index == 0
        assert clusters[0].baseline_score == 100

    def test_merges_via_short_gap_regardless_of_content(self) -> None:
        """gap≤1.5秒は無条件マージ (副信号、盤面変化があっても関係ない)。"""
        t_sec = np.array([0.0, 10.0, 11.5])
        score = np.array([100, 500, 900])
        grids = _empty_grids(3)
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE  # +2個 (設置の署名相当) だが gap<=1.5秒
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 2])
        assert len(clusters) == 1

    def test_separates_when_placement_signature_present(self) -> None:
        """検収基準3b: gap=2.2秒 (v1なら2.5秒以下で誤マージされていた帯域) だが

        設置署名 (+2増加がSEC_PER_HAND超持続) があれば分離を維持する。
        """
        persist = PLACEMENT_SIGNATURE_PERSIST_SEC
        # k=2で+2個のジャンプ開始 (t=1.1)、k=3の時点で経過persist*1.1秒(>persist)で確定。
        t_sec = np.array([0.0, 1.0, 1.1, 1.1 + persist * 1.1, 3.2])
        score = np.array([100, 500, 580, 650, 900])
        grids = _empty_grids(5)
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE
        grids[3, 0, 0] = COLOR_RED
        grids[3, 0, 1] = COLOR_BLUE
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 4])
        assert len(clusters) == 2
        assert [c.fire_index for c in clusters] == [1, 4]

    def test_merges_when_only_transient_plus1_noise(self) -> None:
        """検収基準3c: +1個だけの一時的な写り込みノイズはマージを妨げない。"""
        t_sec = np.array([0.0, 10.0, 10.3, 12.0])
        score = np.array([100, 500, 500, 900])
        grids = _empty_grids(4)
        grids[2, 0, 0] = COLOR_RED  # 中間フレームだけ+1個 (閾値2未満)
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 3])
        assert len(clusters) == 1

    def test_merges_tail_split_across_chain_post_update(self) -> None:
        """検収基準3a: 連鎖後の盤面更新 (ぷよが減るだけ) を挟む長ギャップの尻尾分裂は

        マージされる (v2の「候補j自身の盤面が参照と不一致」だけで分離する
        規則ではここが誤って分離されていた、回帰ガード)。
        """
        t_sec = np.array([0.0, 10.0, 15.0, 20.0])
        score = np.array([100, 27590, 27590, 35274])
        grids = _empty_grids(4)
        grids[0, 0, 0] = COLOR_RED
        grids[0, 0, 1] = COLOR_BLUE
        grids[0, 0, 2] = COLOR_RED  # 開始時点3個
        grids[1] = grids[0].copy()  # 連鎖中は凍結
        # grids[2],[3] (連鎖後更新) は0個 (=減少のみ、増加なし)
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 3])
        assert len(clusters) == 1
        assert clusters[0].fire_index == 3

    def test_gap_exactly_at_threshold_merges(self) -> None:
        """ギャップがちょうど閾値 (FIRE_EVENT_MERGE_GAP_SEC) ならマージする (境界値)。"""
        t_sec = np.array([0.0, 10.0, 10.0 + FIRE_EVENT_MERGE_GAP_SEC])
        score = np.array([100, 500, 900])
        grids = _empty_grids(3)
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE
        clusters = _merge_fire_event_clusters(t_sec, score, grids, [1, 2])
        assert len(clusters) == 1

    def test_three_way_split_merges_into_single_cluster(self) -> None:
        """3分割 (部分加算3回) も1イベントに統合される (連鎖単位クラスタリング)。"""
        t_sec = np.array([0.0, 10.0, 11.0, 12.0])
        score = np.array([100, 300, 600, 1000])
        grids = _empty_grids(4)  # 増加なし
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
        """分裂パターン (部分加算2回、ぷよ数増加なし) は1行に統合される

        (実データ c27.npz 1P game12 と同型: 27590→35274 の分裂が同一9連鎖)。
        """
        t_sec = [0.0, 1.0, 10.0, 12.0]
        score = [0, 20, 27590, 35274]
        grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        fire_rec = _make_record("video_test", "1P", t_sec, score, grids)
        opp_grids = np.zeros((4, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        opp_rec = _make_record("video_test", "2P", t_sec, [0, 0, 0, 0], opp_grids)

        sim = ChainSimulator()
        rows = _process_game(fire_rec, opp_rec, "1P", sim, puyo_q_low=10.0, puyo_q_high=30.0)
        assert len(rows) == 1
        expected_ojama = (35274 - 20) // 70
        assert rows[0]["net_ojama"] == pytest.approx(float(expected_ojama))

    def test_legit_separate_fires_with_placement_signature_yield_two_rows(self) -> None:
        """設置署名 (+2増加が持続) を伴う正当な連続発火は2行に分離される。

        1件目の発火 (index1) と2件目の発火 (index4) の間は gap=2.0秒
        (副信号1.5秒を超える) とし、中間フレームで+2増加が持続することを
        確認する (副信号による無条件マージを回避するため)。
        """
        persist = PLACEMENT_SIGNATURE_PERSIST_SEC
        t_sec = [0.0, 1.0, 1.1, 1.1 + persist * 1.1, 3.0]
        score = [0, 500, 500, 500, 1200]
        grids = np.zeros((5, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        grids[2, 0, 0] = COLOR_RED
        grids[2, 0, 1] = COLOR_BLUE  # +2個 (ジャンプ開始)
        grids[3, 0, 0] = COLOR_RED
        grids[3, 0, 1] = COLOR_BLUE  # 継続 (persist秒超で確定)
        fire_rec = _make_record("video_test", "1P", t_sec, score, grids)
        opp_grids = np.zeros((5, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        opp_rec = _make_record("video_test", "2P", t_sec, [0, 0, 0, 0, 0], opp_grids)

        sim = ChainSimulator()
        rows = _process_game(fire_rec, opp_rec, "1P", sim, puyo_q_low=10.0, puyo_q_high=30.0)
        assert len(rows) == 2
