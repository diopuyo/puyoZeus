"""飽和追従誤差 (saturation_tracking_error) の実データ先行 win-AUC 検証。

定義: saturated_chain_count(飽和連鎖量) - current_max_chain(現在最大連鎖)
    = 「本線完成まであと何連鎖ぶん組む余地があるか」の生値。
    実装ゼロ(既存2列の引き算)。labeled_win.csv の既存列のみで計算する。

検証内容:
    1. 自側生値 (1P/2P それぞれ) の位相別 win-AUC
    2. 相手との差分 (1P-2P) の位相別 win-AUC
    3. 既存指標 (current_max_chain_raw / saturated_chain_count_raw) との
       Pearson相関 (|r|>0.7 なら死票 = 冗長候補)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_saturation_tracking_error
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prescreen_candidates import (
    load_and_pair, eval_candidate,
    TSUMO_EARLY_MAX, TSUMO_LATE_MIN,
    PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE,
    BASELINE_MID_AUC, BASELINE_LATE_AUC,
)

LABELED_CSV = Path("data/indicators_v2/study/labeled_win.csv")
OUT_AUC_CSV = Path("data/indicators_v2/prescreen_sat_track_err_auc.csv")
OUT_CORR_CSV = Path("data/indicators_v2/prescreen_sat_track_err_corr.csv")

EPS = 1.0


def _s(p: pd.DataFrame, col: str, side: str) -> pd.Series:
    key = f"{col}_{side}"
    if key in p.columns:
        return p[key].astype(float)
    return pd.Series(np.nan, index=p.index)


def build_features(paired: pd.DataFrame) -> pd.DataFrame:
    """飽和追従誤差の生値/差分特徴量を構築する。"""
    feats: dict[str, pd.Series] = {}

    sat_1p = _s(paired, "saturated_chain_count_raw", "1p")
    sat_2p = _s(paired, "saturated_chain_count_raw", "2p")
    cur_1p = _s(paired, "current_max_chain_raw", "1p")
    cur_2p = _s(paired, "current_max_chain_raw", "2p")

    err_1p = sat_1p - cur_1p
    err_2p = sat_2p - cur_2p

    feats["sat_track_err_1p"] = err_1p
    feats["sat_track_err_2p"] = err_2p
    feats["sat_track_err_diff"] = err_1p - err_2p
    feats["sat_track_err_ratio"] = err_1p / err_2p.clip(lower=EPS / 100)

    # 参考: 既存2列の diff も併記 (相関比較用)
    feats["current_max_chain_diff"] = cur_1p - cur_2p
    feats["saturated_chain_count_diff"] = sat_1p - sat_2p

    return pd.DataFrame(feats, index=paired.index)


def main() -> None:
    print("=== _tmp_saturation_tracking_error 開始 ===")
    paired = load_and_pair(str(LABELED_CSV))

    won_cols = [c for c in paired.columns if c.startswith("won") and c.endswith("_1p")]
    y = paired[won_cols[0]].astype(float)
    groups = paired["video_id_1p"]
    tcr = paired["tsumo_count_rate_1p"].astype(float)
    phase_masks = {
        PHASE_ALL: pd.Series(True, index=paired.index),
        PHASE_EARLY: tcr <= TSUMO_EARLY_MAX,
        PHASE_MID: (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN),
        PHASE_LATE: tcr > TSUMO_LATE_MIN,
    }
    for ph, m in phase_masks.items():
        print(f"  位相 {ph}: {int(m.sum())} 行")

    feats = build_features(paired)

    print("\n[step1] 位相別 win-AUC (won=1: 1Pが勝つ)")
    auc_rows = []
    eval_cols = [
        "sat_track_err_1p", "sat_track_err_2p",
        "sat_track_err_diff", "sat_track_err_ratio",
    ]
    for col in eval_cols:
        aucs = eval_candidate(feats[col], y, groups, phase_masks)
        auc_rows.append({"candidate": col, **aucs})
    result_df = pd.DataFrame(auc_rows)
    OUT_AUC_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_AUC_CSV, index=False)
    print(result_df.to_string(index=False))
    print(f"[save] {OUT_AUC_CSV}")

    print(f"\n  参考ベースライン: 中盤={BASELINE_MID_AUC}  終盤={BASELINE_LATE_AUC}")

    print("\n[step2] opp_buried/taiou_success 相当ラベルは labeled_win.csv に無し"
          " (won のみで検証、発火イベント単位は exchange_labels 側の候補C4で実施)")

    print("\n[step3] 既存指標との Pearson 相関 (死票判定)")
    corr_rows = []
    corr_pairs = [
        ("sat_track_err_1p", "current_max_chain_raw_1p"),
        ("sat_track_err_1p", "saturated_chain_count_raw_1p"),
        ("sat_track_err_diff", "current_max_chain_diff"),
        ("sat_track_err_diff", "saturated_chain_count_diff"),
    ]
    # current_max_chain_raw_1p 等の実列名を paired 側から引く
    src_map = {
        "current_max_chain_raw_1p": _s(paired, "current_max_chain_raw", "1p"),
        "saturated_chain_count_raw_1p": _s(paired, "saturated_chain_count_raw", "1p"),
        "current_max_chain_diff": feats["current_max_chain_diff"],
        "saturated_chain_count_diff": feats["saturated_chain_count_diff"],
    }
    for cand, existing in corr_pairs:
        x = feats[cand]
        z = src_map[existing]
        mask = x.notna() & z.notna()
        if mask.sum() < 30:
            r = float("nan")
        else:
            r = float(np.corrcoef(x[mask], z[mask])[0, 1])
        flag = "死票候補(|r|>0.7)" if abs(r) > 0.7 else "非冗長"
        corr_rows.append({
            "candidate": cand, "existing_col": existing,
            "pearson_r": r, "n": int(mask.sum()), "judge": flag,
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR_CSV, index=False)
    print(corr_df.to_string(index=False))
    print(f"[save] {OUT_CORR_CSV}")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
