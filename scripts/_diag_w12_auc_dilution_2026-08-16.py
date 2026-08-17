# -*- coding: utf-8 -*-
"""飽和以前の問題: fillna(0)による汚染が単変量AUCをどれだけ薄めているか。
既存 labeled_win_full148.csv (103万行) を読み、ojama_forecast/board_ojama_count
の単変量AUCを (a)全データ(fillna0込み) と (b)ojama_source==0(真値)のみ で比較する。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

CSV = "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
usecols = ["ojama_source", "ojama_forecast", "board_ojama_count", "won"]

chunks = pd.read_csv(CSV, usecols=usecols, chunksize=200000)
dfs = []
for ch in chunks:
    dfs.append(ch)
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["won"])
df["won"] = df["won"].astype(int)

df["ojama_forecast_filled"] = df["ojama_forecast"].fillna(0.0)
df["board_ojama_count_filled"] = df["board_ojama_count"].fillna(0.0)

auc_forecast_all = roc_auc_score(df["won"], df["ojama_forecast_filled"])
auc_ojama_all = roc_auc_score(df["won"], df["board_ojama_count_filled"])

truth = df[df["ojama_source"] == 0.0]
auc_forecast_truth = roc_auc_score(truth["won"], truth["ojama_forecast"])
auc_ojama_truth = roc_auc_score(truth["won"], truth["board_ojama_count"])

print(f"n_all={len(df)} n_truth={len(truth)}")
print(f"ojama_forecast: AUC(全データ,fillna0込み)={auc_forecast_all:.4f} "
      f"AUC(truthのみ)={auc_forecast_truth:.4f}")
print(f"board_ojama_count: AUC(全データ,fillna0込み)={auc_ojama_all:.4f} "
      f"AUC(truthのみ)={auc_ojama_truth:.4f}")
print("符号注記: forecastは値が大きいほど自分に不利=勝率が下がる想定のため "
      "AUC<0.5が正の相関(高いほど負け)を意味する")
