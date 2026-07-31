"""C4検証の較正チェック: taiou_success/returned_competitive 自体が won を予測できるか。

disturbance_rejection が won に効かない(AUC~0.5)原因が
「候補固有の弱さ」か「発火イベント単位ラベル全体がwinと疎結合」かを切り分ける。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prescreen_candidates import eval_candidate

df = pd.read_csv("data/indicators_v2/exchange_labels.csv")
groups = df["video_id"]
phase_masks = {
    "全体": pd.Series(True, index=df.index),
    "序": df["phase"] == "序",
    "中": df["phase"] == "中",
    "終": df["phase"] == "終",
}
y = df["won"].astype(float)
for cand in ["returned", "returned_competitive", "taiou_success", "survived", "opp_buried", "net_ojama"]:
    aucs = eval_candidate(df[cand].astype(float), y, groups, phase_masks)
    print(cand, aucs)
