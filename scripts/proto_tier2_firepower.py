"""tier2 火力コンテキスト指標の試作検証 (CSV オフライン)。

目的: tier1 (安価差分) だけのモデルに、火力ベースの tier2 指標を足すと
  - win 予測 (OOF AUC) が上乗せされるか
  - v29 game B 終盤の誤判定 (手数40付近=満杯盤面で1P誤優勢) が是正されるか
を、study CSV (reach_fire_power 等が既に入っている) だけで安く確認する。

tier2 指標 (いずれも反対称=側入替で符号反転、非線形で生火力差では代替不可):
  - kill_diff  : max(0, 自到達お邪魔 − 相手の受け容量) の 1P−2P。オーバーキル分。
  - expfire_diff: 到達お邪魔 × 発火準備度(min_puyos_to_ignite score) の 1P−2P。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, load_labeled_csv, pair_sides_for_win,
)
from scripts.visualize_advantage_overlay import FEATURES  # noqa: E402

CSV = "data/indicators_v2/study/labeled_win.csv"
GAME_B = (202.0, 283.0)  # v29 game B の書き出し区間 (t_sec)


def _tier2_cols(p: pd.DataFrame) -> pd.DataFrame:
    """paired から tier2 差分特徴を計算して返す。"""
    reach1, reach2 = p["reach_fire_power_raw_1p"], p["reach_fire_power_raw_2p"]
    recv1, recv2 = p["absorption_capacity_raw_1p"], p["absorption_capacity_raw_2p"]
    fire1, fire2 = p["min_puyos_to_ignite_1p"], p["min_puyos_to_ignite_2p"]
    kill1 = (reach1 - recv2).clip(lower=0.0)  # 1Pが2Pをオーバーキルする分
    kill2 = (reach2 - recv1).clip(lower=0.0)
    return pd.DataFrame({
        "kill_diff": (kill1 - kill2).fillna(0.0),
        "expfire_diff": (reach1 * fire1 - reach2 * fire2).fillna(0.0),
    }, index=p.index)


def _base_X(p: pd.DataFrame) -> pd.DataFrame:
    """tier1 安価差分特徴。"""
    return pd.DataFrame(
        {f"{c}_diff": (p[f"{c}_1p"] - p[f"{c}_2p"]).fillna(0.0) for c in FEATURES},
        index=p.index)


def _fit_sym(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    """対称化 (差分反転+ラベル反転) して学習。"""
    Xs = np.vstack([X, -X]); ys = np.concatenate([y, 1 - y])
    m = HistGradientBoostingClassifier(**GBC_PARAMS); m.fit(Xs, ys)
    return m


def _oof_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """GroupKFold(動画単位) の対称化 OOF AUC。"""
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = _fit_sym(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main() -> None:
    df = load_labeled_csv(CSV)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].astype(str).values
    base = _base_X(paired); t2 = _tier2_cols(paired)
    Xb = base.values
    Xt = pd.concat([base, t2], axis=1).values
    print("=== win OOF AUC (対称化, video GroupKFold) ===")
    ab, at = _oof_auc(Xb, y, groups), _oof_auc(Xt, y, groups)
    print(f"  tier1のみ         : {ab:.4f}")
    print(f"  tier1 + 火力tier2 : {at:.4f}")
    print(f"  純増分 ΔAUC       : {at - ab:+.4f}")

    # --- v29 game B 保持検証 (v29除外学習 → game B を採点) ---
    mask_tr = groups != "video_29"
    mb = _fit_sym(Xb[mask_tr], y[mask_tr])
    mt = _fit_sym(Xt[mask_tr], y[mask_tr])
    gb = paired[(paired["video_id_1p"].astype(str) == "video_29")
                & (paired["t_sec_1p"].between(*GAME_B))].sort_values("t_sec_1p")
    if gb.empty:
        print("\n[warn] v29 game B のペア行が見つからない"); return
    bX = _base_X(gb).values
    tX = pd.concat([_base_X(gb), _tier2_cols(gb)], axis=1).values
    adv_b = (mb.predict_proba(bX)[:, 1] - 0.5) * 200
    adv_t = (mt.predict_proba(tX)[:, 1] - 0.5) * 200
    print(f"\n=== v29 game B 保持検証 ({len(gb)}スナップ) ===")
    print("  終盤 (t_rel>=30s) の平均有利不利:")
    trel = gb["t_sec_1p"].values - GAME_B[0]
    late = trel >= 30
    print(f"    tier1のみ         : {adv_b[late].mean():+.1f}")
    print(f"    tier1 + 火力tier2 : {adv_t[late].mean():+.1f}  (2P勝ちなら負が正しい)")
    # 手数40付近 (t_rel~36) の断面
    i = int(np.argmin(np.abs(trel - 36.0)))
    t2v = _tier2_cols(gb).iloc[i]
    print(f"\n  手数40付近 (t_rel={trel[i]:.1f}s) の有利不利:")
    print(f"    tier1のみ         : {adv_b[i]:+.1f}")
    print(f"    tier1 + 火力tier2 : {adv_t[i]:+.1f}")
    print(f"    kill_diff={t2v['kill_diff']:+.1f}  expfire_diff={t2v['expfire_diff']:+.1f}"
          f"  (reach 1P={gb['reach_fire_power_raw_1p'].iloc[i]:.0f}/"
          f"2P={gb['reach_fire_power_raw_2p'].iloc[i]:.0f}, "
          f"受け 1P={gb['absorption_capacity_raw_1p'].iloc[i]:.0f}/"
          f"2P={gb['absorption_capacity_raw_2p'].iloc[i]:.0f})")


if __name__ == "__main__":
    main()
