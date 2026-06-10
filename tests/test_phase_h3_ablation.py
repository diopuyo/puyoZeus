"""scripts.phase_h3_ablation のスモークテスト.

feature 分類ロジック (assign_tiers / build_cumulative_subsets / ranking) の
ユニットテストを提供する。実 CSV を読まず、合成データで挙動を確認する。
"""
from __future__ import annotations

import numpy as np

from scripts.old.phase_h3_ablation import (
    TIER_A_END,
    TIER_B_END,
    TIER_C_END,
    TIER_S_END,
    assign_tiers,
    build_cumulative_subsets,
    compute_permutation_ranking,
    select_recommended_subset,
    video_holdout_split,
)


def _make_synthetic_ds(n: int = 200, d: int = 30, n_videos: int = 6, seed: int = 0) -> dict:
    """合成データセットを生成 (特徴量 0 がラベル決定的)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1).astype(np.int8)
    video_ids = np.array([f"v{i % n_videos:02d}" for i in range(n)])
    time_phases = np.array(["midpoint"] * n)
    feat_cols = [f"f_{i}" for i in range(d)]
    return {
        "X": X, "y": y,
        "video_ids": video_ids, "time_phases": time_phases,
        "feat_cols": feat_cols, "n": n, "d": d,
    }


# ============================
# tier 分類のテスト
# ============================
def test_assign_tiers_correct_partition() -> None:
    """5 tier に重複なく完全分割される."""
    ranking = list(range(280))
    tiers = assign_tiers(ranking, 280)
    assert len(tiers["S"]) == TIER_S_END
    assert len(tiers["A"]) == TIER_A_END - TIER_S_END
    assert len(tiers["B"]) == TIER_B_END - TIER_A_END
    assert len(tiers["C"]) == TIER_C_END - TIER_B_END
    assert len(tiers["D"]) == 280 - TIER_C_END
    flat = sum((tiers[k] for k in "SABCD"), [])
    assert sorted(flat) == list(range(280))


def test_assign_tiers_handles_small_total() -> None:
    """total < TIER_C_END でも例外を出さない."""
    ranking = list(range(50))
    tiers = assign_tiers(ranking, 50)
    assert len(tiers["S"]) == TIER_S_END
    assert len(tiers["A"]) == TIER_A_END - TIER_S_END
    assert len(tiers["B"]) == 0
    assert len(tiers["C"]) == 0
    assert len(tiers["D"]) == 0


# ============================
# cumulative subset のテスト
# ============================
def test_build_cumulative_subsets_growing() -> None:
    """累積部分集合のサイズは単調増加."""
    ranking = list(range(280))
    tiers = assign_tiers(ranking, 280)
    subsets = build_cumulative_subsets(tiers)
    keys = list(subsets.keys())
    assert keys == ["S", "S+A", "S+A+B", "S+A+B+C", "S+A+B+C+D"]
    sizes = [len(subsets[k]) for k in keys]
    assert sizes == sorted(sizes)
    assert sizes[0] == TIER_S_END
    assert sizes[-1] == 280


def test_build_cumulative_subsets_inclusion() -> None:
    """前段の subset が後段に完全に含まれる."""
    ranking = list(range(280))
    tiers = assign_tiers(ranking, 280)
    subsets = build_cumulative_subsets(tiers)
    keys = list(subsets.keys())
    for prev, nxt in zip(keys, keys[1:]):
        assert set(subsets[prev]).issubset(set(subsets[nxt]))


# ============================
# ranking のスモークテスト
# ============================
def test_permutation_ranking_smoke() -> None:
    """合成データで feature 0 が top にランクされる (signal)."""
    ds = _make_synthetic_ds(n=200, d=10, n_videos=4, seed=42)
    train_mask, test_mask = video_holdout_split(ds["video_ids"], n_test=2, seed=0)
    ranking = compute_permutation_ranking(ds, train_mask, test_mask)
    # ranking の長さ = 特徴量数
    assert len(ranking) == ds["d"]
    # 全 index がユニーク
    assert len(set(ranking)) == ds["d"]
    # signal feature (index=0) は top 3 以内であってほしい
    assert ranking.index(0) < 3


# ============================
# select_recommended_subset のテスト
# ============================
def test_select_recommended_subset_picks_best() -> None:
    """LR + LR phase avg の合計最大の subset を選ぶ."""
    fake = {
        "S": {"n_features": 20, "lr_video_holdout": 0.7, "lr_phase_avg": 0.6,
              "hgbt_video_holdout": 0.6, "hgbt_phase_avg": 0.6},
        "S+A": {"n_features": 50, "lr_video_holdout": 0.75, "lr_phase_avg": 0.65,
                "hgbt_video_holdout": 0.6, "hgbt_phase_avg": 0.6},
        "S+A+B": {"n_features": 100, "lr_video_holdout": 0.72, "lr_phase_avg": 0.66,
                  "hgbt_video_holdout": 0.6, "hgbt_phase_avg": 0.6},
    }
    key, val = select_recommended_subset(fake)
    assert key == "S+A"
    assert val["n_features"] == 50


# ============================
# video_holdout_split のテスト
# ============================
def test_video_holdout_split_no_leak() -> None:
    """train/test に同じ動画 ID は出現しない."""
    ds = _make_synthetic_ds(n=120, d=8, n_videos=6, seed=0)
    train_mask, test_mask = video_holdout_split(ds["video_ids"], n_test=2, seed=0)
    assert train_mask.sum() + test_mask.sum() == ds["n"]
    train_videos = set(ds["video_ids"][train_mask])
    test_videos = set(ds["video_ids"][test_mask])
    assert train_videos.isdisjoint(test_videos)
