"""augment_exchange_labels_with_sim.py の並列化 (--workers、2026-08-02) の回帰テスト。

軽量な合成 npz ファイル (tmp_path) のみを使用する。以下2観点を担保する:
    1. balance_videos_by_event_count (割当バランサ) の単体テスト
    2. --workers=1 (逐次) と --workers>1 (並列) の出力が完全一致すること
       (sim_* 3列、NaN含む)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.board import BOARD_COLS, BOARD_ROWS
from scripts.augment_exchange_labels_with_sim import (
    SIM_NAN,
    _run_parallel,
    _run_sequential,
    balance_videos_by_event_count,
)


# =============================================================================
# balance_videos_by_event_count (割当バランサ)
# =============================================================================

class TestBalanceVideosByEventCount:
    def test_distributes_evenly_when_counts_equal(self) -> None:
        counts = {"a": 10, "b": 10, "c": 10, "d": 10}
        assigned = balance_videos_by_event_count(counts, n_workers=2)
        loads = [sum(counts[v] for v in vids) for vids in assigned]
        assert loads == [20, 20]

    def test_largest_video_goes_to_first_empty_worker(self) -> None:
        counts = {"big": 100, "small1": 5, "small2": 5}
        assigned = balance_videos_by_event_count(counts, n_workers=2)
        assert "big" in assigned[0]

    def test_greedy_balances_skewed_counts(self) -> None:
        # 大1本+小4本 -> 大1本を1worker、小4本をもう1workerに寄せてバランスさせる。
        counts = {"big": 40, "s1": 10, "s2": 10, "s3": 10, "s4": 10}
        assigned = balance_videos_by_event_count(counts, n_workers=2)
        loads = sorted(sum(counts[v] for v in vids) for vids in assigned)
        assert loads == [40, 40]

    def test_more_workers_than_videos_leaves_empty_lists(self) -> None:
        counts = {"a": 5, "b": 3}
        assigned = balance_videos_by_event_count(counts, n_workers=5)
        assert len(assigned) == 5
        non_empty = [vids for vids in assigned if vids]
        assert len(non_empty) == 2

    def test_all_videos_are_assigned_exactly_once(self) -> None:
        counts = {f"v{i}": i + 1 for i in range(9)}
        assigned = balance_videos_by_event_count(counts, n_workers=3)
        all_assigned = sorted(v for vids in assigned for v in vids)
        assert all_assigned == sorted(counts.keys())

    def test_empty_counts_returns_all_empty_lists(self) -> None:
        assigned = balance_videos_by_event_count({}, n_workers=4)
        assert assigned == [[], [], [], []]


# =============================================================================
# --workers=1 (逐次) と --workers>1 (並列) の出力一致
# =============================================================================

def _write_synthetic_npz(path, video_id: str, n_frames: int, seed: int) -> None:
    """_load_npz が読める最小スキーマの合成 npz ファイルを書き出す。"""
    rng = np.random.default_rng(seed)
    sides = np.array((["1P"] * n_frames) + (["2P"] * n_frames))
    video_ids = np.array([video_id] * (2 * n_frames))
    t_sec = np.concatenate([np.arange(n_frames, dtype=np.float32) * 3.0] * 2)
    game_idx = np.zeros(2 * n_frames, dtype=np.int32)
    grids = rng.integers(0, 2, size=(2 * n_frames, BOARD_ROWS, BOARD_COLS)).astype(np.int8)
    won = np.zeros(2 * n_frames, dtype=np.float32)
    # 発火を模擬 (SCORE_DELTA_FIRE=80超のジャンプを複数回入れる)
    score = np.tile(np.cumsum(rng.integers(50, 300, size=n_frames)).astype(np.int32), 2)
    np.savez(path, video_id=video_ids, side=sides, t_sec=t_sec, game_idx=game_idx,
             grids=grids, won=won, score=score)


def _make_events_df(video_ids: list[str], n_events_each: int) -> pd.DataFrame:
    """各動画に n_events_each 件ずつの発火イベント行を持つ合成 DataFrame。"""
    rows = []
    for vid in video_ids:
        for i in range(n_events_each):
            rows.append({
                "video_id": vid, "game_idx": 0, "t_sec": float(3 + i * 6),
                "fire_side": "1P" if i % 2 == 0 else "2P",
                "approx_fire_chains": float(1 + i % 5),
            })
    return pd.DataFrame(rows)


class TestSequentialVsParallelEquivalence:
    """workers=1 と workers>1 で出力 (sim_* 3列、NaN含む) が完全一致するか。"""

    def _setup(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        video_ids = ["c901", "c902", "c903", "c904"]
        for i, vid in enumerate(video_ids):
            _write_synthetic_npz(npz_dir / f"{vid}.npz", f"video_{vid}", n_frames=30, seed=100 + i)
        df = _make_events_df([f"video_{v}" for v in video_ids], n_events_each=5)
        return npz_dir, df

    def test_workers_1_vs_4_produce_identical_output(self, tmp_path) -> None:
        npz_dir, df = self._setup(tmp_path)
        k1, e1, d1 = _run_sequential(df.copy(), npz_dir, mode="precise")
        k4, e4, d4 = _run_parallel(df.copy(), npz_dir, mode="precise", n_workers=4)

        assert np.array_equal(np.isnan(k1), np.isnan(k4))
        assert np.array_equal(np.isnan(e1), np.isnan(e4))
        assert np.array_equal(np.isnan(d1), np.isnan(d4))
        valid = ~np.isnan(d1)
        assert np.allclose(k1[valid], k4[valid])
        assert np.allclose(e1[valid], e4[valid])
        assert np.allclose(d1[valid], d4[valid])

    def test_sequential_run_twice_is_reproducible(self, tmp_path) -> None:
        """--workers=1 を2回実行しても同一結果になる (MCシードが盤面決定論的か確認)。"""
        npz_dir, df = self._setup(tmp_path)
        k_a, e_a, d_a = _run_sequential(df.copy(), npz_dir, mode="precise")
        k_b, e_b, d_b = _run_sequential(df.copy(), npz_dir, mode="precise")
        assert np.array_equal(k_a, k_b, equal_nan=True)
        assert np.array_equal(e_a, e_b, equal_nan=True)
        assert np.array_equal(d_a, d_b, equal_nan=True)

    def test_workers_more_than_videos_still_matches_sequential(self, tmp_path) -> None:
        """動画本数より worker 数が多い場合 (空タスクが生じる) でも一致すること。"""
        npz_dir, df = self._setup(tmp_path)
        k1, e1, d1 = _run_sequential(df.copy(), npz_dir, mode="precise")
        k8, e8, d8 = _run_parallel(df.copy(), npz_dir, mode="precise", n_workers=8)
        assert np.array_equal(k1, k8, equal_nan=True)
        assert np.array_equal(e1, e8, equal_nan=True)
        assert np.array_equal(d1, d8, equal_nan=True)
