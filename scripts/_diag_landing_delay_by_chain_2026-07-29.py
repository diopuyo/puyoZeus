"""読み取り専用の集計診断: 連鎖数別の着弾遅延実測 (2026-07-29)。

user指摘: `scripts/measure_exchange_effectiveness.py` の着弾遅延2アンカー線形
補間 (8連鎖=14.5秒, 13連鎖=18秒) を短い連鎖 (1〜4) に外挿すると切片8.9秒が
残ってしまい物理的に破綻する。既存データを集計し、連鎖数別の実測遅延分布・
比例モデル対線形モデルの当てはまり・短連鎖nの実在有無を報告する。

制約: src/indicators_v2.py, src/chain_bitboard.py,
scripts/measure_exchange_effectiveness.py は一切変更しない (読むだけ)。
認識の再実行もしない (既存CSVの集計のみ)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent

REGEN_CSV = PROJ_ROOT / "data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv"
OLD_CSV = PROJ_ROOT / "data/indicators_v2/exchange_landing_delay.csv"
CHAIN_ANIM_BINS_CSV = (
    PROJ_ROOT
    / "data/verify/recognition_diag_chain_anim_duration_multi/chain_count_bins.csv"
)
CHAIN_ANIM_MODEL_JSON = (
    PROJ_ROOT / "data/verify/recognition_diag_chain_anim_duration_multi/model_fit.json"
)

# 現行モデル (scripts/measure_exchange_effectiveness.py の値をそのまま転記、
# 読むだけで再定義。呼び出しはしない=依存を作らない)。
CURRENT_ANCHOR_LOW_CHAIN = 8
CURRENT_ANCHOR_LOW_SEC = 14.5
CURRENT_ANCHOR_HIGH_CHAIN = 13
CURRENT_ANCHOR_HIGH_SEC = 18.0
CURRENT_SLOPE = (CURRENT_ANCHOR_HIGH_SEC - CURRENT_ANCHOR_LOW_SEC) / (
    CURRENT_ANCHOR_HIGH_CHAIN - CURRENT_ANCHOR_LOW_CHAIN
)
CURRENT_INTERCEPT = CURRENT_ANCHOR_LOW_SEC - CURRENT_SLOPE * CURRENT_ANCHOR_LOW_CHAIN

SEC_PER_HAND = 0.733  # src/indicators_v2.py:SEC_PER_HAND (転記、読むだけ)

BIN_EDGES = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 7, "5-7"),
    (8, 12, "8-12"),
    (13, 999, "13+"),
]


def load_landed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    landed = df[df["detection_status"] == "landed"].copy()
    landed = landed.dropna(subset=["delay_from_chain_start_sec", "chain_count"])
    return landed


def bin_label(chain_count: int) -> str:
    for lo, hi, label in BIN_EDGES:
        if lo <= chain_count <= hi:
            return label
    return "13+"


def summarize(landed: pd.DataFrame, delay_col: str) -> pd.DataFrame:
    landed = landed.copy()
    landed["bin"] = landed["chain_count"].apply(bin_label)
    order = [b[2] for b in BIN_EDGES]
    rows = []
    for label in order:
        sub = landed[landed["bin"] == label][delay_col]
        if len(sub) == 0:
            rows.append(
                {"連鎖数": label, "n": 0, "中央値": np.nan, "平均": np.nan,
                 "最小": np.nan, "最大": np.nan}
            )
            continue
        rows.append(
            {
                "連鎖数": label,
                "n": len(sub),
                "中央値": round(float(sub.median()), 2),
                "平均": round(float(sub.mean()), 2),
                "最小": round(float(sub.min()), 2),
                "最大": round(float(sub.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def fit_models(landed: pd.DataFrame, delay_col: str) -> dict:
    """比例モデル(原点通過) と現行2アンカー線形モデルの残差を比較する。"""
    x = landed["chain_count"].to_numpy(dtype=float)
    y = landed[delay_col].to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0}

    # 比例モデル a*x (最小二乗、原点通過): a = sum(xy)/sum(x^2)
    a_prop = float(np.sum(x * y) / np.sum(x * x))
    pred_prop = a_prop * x
    resid_prop = y - pred_prop

    # 現行モデル (固定係数、当てはめではなく実データに対する残差評価のみ)
    pred_current = CURRENT_INTERCEPT + CURRENT_SLOPE * x
    resid_current = y - pred_current

    def rmse(r: np.ndarray) -> float:
        return float(np.sqrt(np.mean(r ** 2)))

    def mae(r: np.ndarray) -> float:
        return float(np.mean(np.abs(r)))

    return {
        "n": n,
        "proportional_a": round(a_prop, 4),
        "proportional_rmse": round(rmse(resid_prop), 3),
        "proportional_mae": round(mae(resid_prop), 3),
        "current_intercept": round(CURRENT_INTERCEPT, 4),
        "current_slope": round(CURRENT_SLOPE, 4),
        "current_rmse": round(rmse(resid_current), 3),
        "current_mae": round(mae(resid_current), 3),
    }


def main() -> None:
    print("=" * 70)
    print("1. データ存在確認")
    print("=" * 70)
    for path in (REGEN_CSV, OLD_CSV):
        exists = path.exists()
        print(f"  {path.name}: exists={exists}")

    regen_all = pd.read_csv(REGEN_CSV)
    old_all = pd.read_csv(OLD_CSV)
    print(f"\n  regen 総行数={len(regen_all)}  detection_status内訳:")
    print(regen_all["detection_status"].value_counts().to_string())
    print(f"\n  old 総行数={len(old_all)}  detection_status内訳:")
    print(old_all["detection_status"].value_counts().to_string())

    regen_landed = load_landed(REGEN_CSV)
    old_landed = load_landed(OLD_CSV)
    print(f"\n  regen landed n={len(regen_landed)} (delay_from_chain_start_sec 欠損なし)")
    print(f"  old landed n={len(old_landed)} (delay_from_chain_start_sec 欠損なし)")

    print()
    print("=" * 70)
    print("2. 連鎖数別 着弾遅延 (delay_from_chain_start_sec) -- regen (post-fix)")
    print("=" * 70)
    print(summarize(regen_landed, "delay_from_chain_start_sec").to_string(index=False))

    print()
    print("=" * 70)
    print("3. 連鎖数別 着弾遅延 (delay_from_chain_start_sec) -- old (pre-fix, 参考)")
    print("=" * 70)
    print(summarize(old_landed, "delay_from_chain_start_sec").to_string(index=False))

    print()
    print("=" * 70)
    print("3b. 参考: delay_sec (t_fire基準、誤って過小評価しやすい列) regen")
    print("=" * 70)
    print(summarize(regen_landed, "delay_sec").to_string(index=False))

    print()
    print("=" * 70)
    print("4. 短連鎖(1-4)の実測n数")
    print("=" * 70)
    for label in ["1", "2", "3", "4"]:
        n_regen = (regen_landed["chain_count"].apply(bin_label) == label).sum()
        n_old = (old_landed["chain_count"].apply(bin_label) == label).sum()
        print(f"  連鎖{label}: regen n={n_regen}, old n={n_old}")

    print()
    print("=" * 70)
    print("5. モデル当てはめ比較 (比例モデル a*chain vs 現行2アンカー線形)")
    print("=" * 70)
    print("  -- regen landed (n=%d) --" % len(regen_landed))
    print(json.dumps(fit_models(regen_landed, "delay_from_chain_start_sec"), ensure_ascii=False, indent=2))
    print("  -- old landed (n=%d, 参考) --" % len(old_landed))
    print(json.dumps(fit_models(old_landed, "delay_from_chain_start_sec"), ensure_ascii=False, indent=2))
    combined = pd.concat([regen_landed, old_landed], ignore_index=True)
    print("  -- combined regen+old (n=%d, 重複試合を含みうる参考値) --" % len(combined))
    print(json.dumps(fit_models(combined, "delay_from_chain_start_sec"), ensure_ascii=False, indent=2))

    print()
    print("=" * 70)
    print("6. 代替データ: 連鎖アニメ総時間 (視覚的settle実測, 23動画 n=418)")
    print("   src/recognition_pipeline.py CHAIN_HOLD_BASE_SEC/PER_STEP_SEC 較正元")
    print("=" * 70)
    if CHAIN_ANIM_BINS_CSV.exists():
        anim_bins = pd.read_csv(CHAIN_ANIM_BINS_CSV)
        print(anim_bins[["chain_bin", "n", "visual_median", "visual_mean", "visual_min", "visual_max"]].to_string(index=False))
    else:
        print("  ファイルなし")
    if CHAIN_ANIM_MODEL_JSON.exists():
        model_fit = json.loads(CHAIN_ANIM_MODEL_JSON.read_text(encoding="utf-8"))
        print("\n  線形フィット(視覚的settle, erase animのみ):", json.dumps(model_fit, ensure_ascii=False))

    print()
    print("=" * 70)
    print("7. 推奨モデルでの 連鎖数1-13 推定手数表")
    print("=" * 70)

    def hands_current(chain: int) -> tuple[float, int]:
        delay = max(0.0, CURRENT_INTERCEPT + CURRENT_SLOPE * chain)
        return delay, int(delay // SEC_PER_HAND)

    print(f"  現行モデル: delay = {CURRENT_INTERCEPT:.3f} + {CURRENT_SLOPE:.3f}*chain")
    for c in range(1, 14):
        d, h = hands_current(c)
        print(f"    連鎖{c:2d}: delay={d:6.2f}s hands={h}")


if __name__ == "__main__":
    main()
