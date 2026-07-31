"""最終確認: 本番実装(バグ修正後)の near_future_fire_k1-5 vs current_max_chain。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.model_indicator_win import (
    TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO, pair_sides_for_win, build_features,
)

df = pd.read_csv("data/indicators_v2/study/near_future_prod_verify_result.csv")
df = df.dropna(subset=["video_id", "side", "won"]).copy()
df["won"] = df["won"].astype(int)
paired = pair_sides_for_win(df, 1.0)
y = paired["won_1p"].astype(int).values
tsumo = paired["tsumo_1p"].astype(float).values
q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
masks = {"序盤": tsumo <= q33, "中盤": (tsumo > q33) & (tsumo <= q67), "終盤": tsumo > q67}


def diff_auc(col, mask=None):
    feat = build_features(paired, [col])
    score = feat[col + "_diff"].fillna(0.0).values
    yy = y
    if mask is not None:
        score, yy = score[mask], y[mask]
    auc = roc_auc_score(yy, score)
    return max(auc, 1 - auc)


cols = ["current_max_chain_raw"] + [f"near_future_fire_k{k}_raw" for k in range(1, 6)]
base = {}
header = "指標".ljust(28) + "全体".rjust(8) + "序盤".rjust(8) + "中盤".rjust(8) + "終盤".rjust(8)
print(header)
results = {}
for c in cols:
    row = {"全体": diff_auc(c)}
    for p, m in masks.items():
        row[p] = diff_auc(c, m)
    if c == "current_max_chain_raw":
        base = row
    results[c] = row
    line = c.ljust(28)
    for p in ["全体", "序盤", "中盤", "終盤"]:
        line += f"{row[p]:>8.4f}"
    print(line)

print()
print("--- current_max_chain比の差分(本番実装、バグ修正後) ---")
for c in cols[1:]:
    line = c.ljust(28)
    for p in ["全体", "序盤", "中盤", "終盤"]:
        line += f"{results[c][p] - base[p]:>+8.4f}"
    print(line)
