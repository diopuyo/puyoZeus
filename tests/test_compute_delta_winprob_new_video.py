"""scripts/compute_delta_winprob_new_video.py (併用スタッキング版) のテスト。

2026-08-03 修正A (案D単体フォールバック廃止 → 併用スタッキング + sim_*その場計算)
の検収: sim列その場計算の突合 (既存66動画の事前計算済み aug CSV 値と、
その場で `_compute_sim_columns_for_row` を呼んだ結果が一致すること)。
実データ (data/verify/..., data/indicators_v2/...) が無い環境では skip する。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.augment_exchange_labels_with_sim import _load_video_cache
from scripts.compute_delta_winprob_new_video import SIM_MODE, _build_features
from src.exchange_predictor import ExchangeModelBundle

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv")
NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")

pytestmark = pytest.mark.skipif(
    not (AUG_CSV.exists() and NPZ_DIR.exists()),
    reason="実データ (aug CSV / npz) が無い環境のためskip",
)


def _load_sample_rows(video_id: str = "video_c61", n: int = 5) -> pd.DataFrame:
    df = pd.read_csv(AUG_CSV)
    sub = df.loc[df["video_id"] == video_id].head(n).reset_index(drop=True)
    assert len(sub) > 0, f"{video_id} の行が aug CSV に無い"
    return sub


class TestSimColumnOnTheFlyParity:
    """指摘: sim列その場計算の突合 (既存66動画の事前計算値と一致すること)。"""

    def test_on_the_fly_sim_matches_precomputed_aug_csv(self) -> None:
        from scripts.augment_exchange_labels_with_sim import _compute_sim_columns_for_row

        rows = _load_sample_rows()
        cache = _load_video_cache("video_c61", NPZ_DIR)
        assert cache is not None
        mismatches = []
        for _, row in rows.iterrows():
            k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, SIM_MODE)
            if not (
                k_hands == pytest.approx(row["sim_k_hands"])
                and exp_counter == pytest.approx(row["sim_expected_counter_ojama"], abs=1e-6)
                and damage == pytest.approx(row["sim_damage_score"], abs=1e-6)
            ):
                mismatches.append((row["t_sec"], row["fire_side"],
                                   (k_hands, exp_counter, damage),
                                   (row["sim_k_hands"], row["sim_expected_counter_ojama"],
                                    row["sim_damage_score"])))
        assert not mismatches, f"その場計算が事前計算値と不一致: {mismatches}"


class TestBuildFeaturesIncludesSimCols:
    def test_build_features_appends_sim_values(self) -> None:
        model = ExchangeModelBundle(
            cls_model=None, reg_model=None,
            indicator_bases=["current_max_chain"], feature_names=[],
            phases=("序", "中", "終"), fire_sides=("1P", "2P"), metadata={},
            sim_feature_cols=("sim_k_hands", "sim_expected_counter_ojama", "sim_damage_score"),
        )
        row = pd.Series({
            "phase": "中", "fire_side": "1P",
            "fire_current_max_chain": 1.0, "opp_current_max_chain": 2.0, "diff_current_max_chain": -1.0,
        })
        features = _build_features(row, model, (3.0, 150.0, 0.4))
        assert features["sim_k_hands"] == 3.0
        assert features["sim_expected_counter_ojama"] == 150.0
        assert features["sim_damage_score"] == 0.4
