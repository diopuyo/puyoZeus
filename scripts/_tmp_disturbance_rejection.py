"""候補C4 外乱除去比 (disturbance_rejection) の実データ先行 AUC 検証。

定義: disturbance_rejection = returned(counter_ojama) / incoming(attack_ojama)
    「受けた予告お邪魔のうち相殺して打ち返せた割合」= お邪魔会計の実効防御性能。
    発火イベント単位 (exchange_labels.csv)。

実装:
    exchange_labels.csv には net_ojama (= attack_ojama - counter_ojama,
    return_window_sec 窓) と net_ojama_after (同various T_guard窓) は
    既にあるが attack_ojama 自体の生値列が無いため、boards_lean_fixed npz の
    score 系列から label_exchange_outcome.py と全く同じロジック(既存関数を
    そのまま再利用)で delta_score を再検出し attack_ojama を復元する。
    counter_ojama = attack_ojama - net_ojama (既存列から逆算、追加simゼロ)。

検証:
    1. disturbance_rejection (return_window版 / T_guard版) の位相別 AUC
       (target: won / opp_buried / taiou_success)
    2. 既存お邪魔会計列 (returned, returned_competitive, net_ojama,
       net_ojama_after) との Pearson相関 (死票判定)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_disturbance_rejection
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

from scripts.label_exchange_outcome import _load_npz, NPZ_DIR  # noqa: E402
from src.scoring import OJAMA_RATE_STANDARD  # noqa: E402
from scripts.prescreen_candidates import eval_candidate  # noqa: E402

EXCHANGE_CSV = Path("data/indicators_v2/exchange_labels.csv")
OUT_AUC_CSV = Path("data/indicators_v2/prescreen_disturbance_rejection_auc.csv")
OUT_CORR_CSV = Path("data/indicators_v2/prescreen_disturbance_rejection_corr.csv")

T_MATCH_TOL: float = 0.05  # 秒 (同フレーム一致判定の許容誤差)
EPS: float = 1.0


def _recover_attack_ojama(df: pd.DataFrame) -> pd.Series:
    """exchange_labels.csv の各行に対応する attack_ojama (delta_score//70) を
    boards_lean_fixed npz の score 系列から再検出して復元する。

    label_exchange_outcome.py の _process_game と全く同じ
    「直前有効scoreとの差分」ロジックを踏襲 (delta_score 自体は
    exchange_labels.csv に保存されていないため)。
    """
    attack_ojama = pd.Series(np.nan, index=df.index, dtype=float)
    n_unmatched = 0

    for vid, sub in df.groupby("video_id"):
        short = vid.replace("video_", "")
        npz_path = NPZ_DIR / f"{short}.npz"
        if not npz_path.exists():
            print(f"  [WARN] npz見つからず: {npz_path} ({len(sub)} 行スキップ)")
            n_unmatched += len(sub)
            continue
        records = _load_npz(npz_path)
        by_side = {r.side: r for r in records}

        for idx, row in sub.iterrows():
            side = row["fire_side"]
            if side not in by_side:
                n_unmatched += 1
                continue
            rec = by_side[side]
            game_mask = rec.game_idx == int(row["game_idx"])
            if not game_mask.any():
                n_unmatched += 1
                continue
            t_arr = rec.t_sec[game_mask]
            score_arr = rec.score[game_mask]
            local_idx = np.where(game_mask)[0]
            diffs = np.abs(t_arr - float(row["t_sec"]))
            best = int(diffs.argmin())
            if diffs[best] > T_MATCH_TOL:
                n_unmatched += 1
                continue
            fi = local_idx[best]
            s_fire = int(rec.score[fi])
            prev_valid = -1
            for j in range(fi - 1, -1, -1):
                if rec.score[j] >= 0:
                    prev_valid = int(rec.score[j])
                    break
            if prev_valid < 0 or s_fire < 0:
                n_unmatched += 1
                continue
            delta = s_fire - prev_valid
            attack_ojama.loc[idx] = float(max(0, delta) // OJAMA_RATE_STANDARD)

    print(f"  attack_ojama 復元: {attack_ojama.notna().sum()} / {len(df)} 行"
          f" (不一致/欠損 {n_unmatched} 行)")
    return attack_ojama


def main() -> None:
    print("=== _tmp_disturbance_rejection 開始 ===")
    df = pd.read_csv(EXCHANGE_CSV)
    print(f"[load] {EXCHANGE_CSV}: {df.shape}")

    print("\n[step1] attack_ojama (incoming) 復元 ...")
    df["attack_ojama_raw"] = _recover_attack_ojama(df)

    valid = df["attack_ojama_raw"].notna() & (df["attack_ojama_raw"] > 0)
    print(f"  attack_ojama>0 の有効行: {int(valid.sum())} / {len(df)}")
    df = df[valid].copy()

    # counter_ojama = attack_ojama - net_ojama (同じ return_window_sec 窓)
    df["counter_ojama_raw"] = df["attack_ojama_raw"] - df["net_ojama"]
    # T_guard版 (net_ojama_after は同じ attack_ojama 定義、窓だけ違う)
    df["counter_ojama_after_raw"] = df["attack_ojama_raw"] - df["net_ojama_after"]

    df["disturbance_rejection"] = (
        df["counter_ojama_raw"] / df["attack_ojama_raw"].clip(lower=EPS)
    )
    df["disturbance_rejection_tguard"] = (
        df["counter_ojama_after_raw"] / df["attack_ojama_raw"].clip(lower=EPS)
    )

    print(f"  disturbance_rejection 統計: mean={df['disturbance_rejection'].mean():.3f}"
          f" median={df['disturbance_rejection'].median():.3f}"
          f" std={df['disturbance_rejection'].std():.3f}")
    print(f"  disturbance_rejection_tguard 統計: "
          f"mean={df['disturbance_rejection_tguard'].mean():.3f}"
          f" median={df['disturbance_rejection_tguard'].median():.3f}")

    groups = df["video_id"]
    phase_masks = {
        "全体": pd.Series(True, index=df.index),
        "序": df["phase"] == "序",
        "中": df["phase"] == "中",
        "終": df["phase"] == "終",
    }
    for ph, m in phase_masks.items():
        print(f"  位相 {ph}: {int(m.sum())} 行")

    targets = {
        "won": df["won"].astype(float),
        "opp_buried": df["opp_buried"].astype(float),
        "taiou_success": df["taiou_success"].astype(float),
        "survived": df["survived"].astype(float),
    }
    cand_cols = ["disturbance_rejection", "disturbance_rejection_tguard"]

    print("\n[step2] 位相別 win/opp_buried/taiou_success AUC ...")
    auc_rows = []
    for cand in cand_cols:
        for tgt_name, y in targets.items():
            aucs = eval_candidate(df[cand], y, groups, phase_masks)
            auc_rows.append({"candidate": cand, "target": tgt_name, **aucs})
    result_df = pd.DataFrame(auc_rows)
    OUT_AUC_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_AUC_CSV, index=False)
    print(result_df.to_string(index=False))
    print(f"[save] {OUT_AUC_CSV}")

    print("\n[step3] 既存お邪魔会計列との Pearson相関 (死票判定)")
    existing_cols = [
        "returned", "returned_competitive", "net_ojama", "net_ojama_after",
        "taiou_success",
    ]
    corr_rows = []
    for cand in cand_cols:
        for existing in existing_cols:
            x = df[cand].astype(float)
            z = df[existing].astype(float)
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
