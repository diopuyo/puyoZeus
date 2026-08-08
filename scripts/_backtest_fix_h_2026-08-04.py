"""修正H (防御側推定器の格上げ) 全域バックテスト (2026-08-04)。

18,057イベントで「受け切れ判定」(=severity>0、旧二値と同一の発動基準) を
予測とみなし、受け側の実窒息 (opp_buried、既存ラベル再利用) との
precision/recall + 校正曲線を、修正前 (realizable_counter=expected_fire_
power単体) と修正後 (realizable_counter=max(immediate_fire_power,
expected_fire_power)、修正H) で比較する。

既存資産の再利用のみ: reconstruct_event_board_pair (盤面復元)、
board_room (空き容量)、sim_expected_counter_ojama 列 (aug CSV、既に計算済み
のexpected_fire_power、MC計算の重い部分は再計算しない) をベースにし、
immediate_fire_power (決定論的・軽量、既存指標III-2) のみ新規に計算して
max を取る (修正Hの定義そのまま、_realizable_counter_ojama と同一ロジック
だが重いexpected_fire_powerの再計算を避けるための構成)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.indicators_v2 as iv
from scripts.compute_exchange_delta_winprob import OJAMA_MAX_DROP_PER_TURN, _load_video_npz, reconstruct_event_board_pair
from scripts.visualize_advantage_overlay import board_room

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/fix_h_backtest_2026-08-04")
SEVERITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def compute_room_and_immediate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """行ごとに room と immediate_fire_power (決定論的・軽量) を npz から
    計算する (video単位でキャッシュ再利用、既存 reconstruct_event_board_pair
    再利用、expected_fire_power の重いMC再計算はしない)。
    """
    room = np.full(len(df), np.nan)
    immediate = np.full(len(df), np.nan)
    for video_id, sub in df.groupby("video_id"):
        stem = video_id[len("video_"):] if video_id.startswith("video_") else video_id
        cache = _load_video_npz(stem, REGEN_NPZ_DIR)
        if cache is None:
            continue
        for idx, row in sub.iterrows():
            pair = reconstruct_event_board_pair(
                cache, int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"]))
            if pair is None:
                continue
            _fire_board, opp_board = pair
            i = df.index.get_loc(idx)
            room[i] = board_room(opp_board)
            immediate[i] = iv.immediate_fire_power(opp_board).raw
    return room, immediate


def severity_flag(pending: np.ndarray, counter: np.ndarray, room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_excess = pending - counter - room
    severity = np.clip(raw_excess / OJAMA_MAX_DROP_PER_TURN, 0.0, 1.0)
    threatened = pending > 0.0
    flag = np.where(threatened & (raw_excess > 0.0), 1.0, 0.0)
    severity = np.where(threatened, severity, 0.0)
    return flag, severity


def precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = float(((pred == 1) & (truth == 1)).sum())
    fp = float(((pred == 1) & (truth == 0)).sum())
    fn = float(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"n_pred_positive": int(tp + fp), "n_true_positive": int(tp + fn),
            "precision": precision, "recall": recall}


def calibration_by_severity_bin(severity: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(SEVERITY_BINS[:-1], SEVERITY_BINS[1:]):
        mask = (severity >= lo) & (severity < hi if hi < 1.0 else severity <= hi)
        n = int(mask.sum())
        rate = float(truth[mask].mean()) if n > 0 else float("nan")
        rows.append({"severity_bin": f"[{lo:.1f},{hi:.1f}{']' if hi >= 1.0 else ')'}", "n": n, "opp_buried_rate": rate})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(AUG_CSV)
    print(f"=== 入力: {len(df)}行 ===")

    room, immediate = compute_room_and_immediate(df)
    n_ok = int(np.sum(~np.isnan(room)))
    print(f"盤面復元成功: {n_ok}/{len(df)} ({n_ok / len(df):.1%})")

    valid = ~np.isnan(room)
    df_v = df.loc[valid].reset_index(drop=True)
    room_v, immediate_v = room[valid], immediate[valid]
    pending = df_v["net_ojama_after"].astype(float).values
    counter_old_v = df_v["sim_expected_counter_ojama"].astype(float).values
    counter_new_v = np.maximum(counter_old_v, immediate_v)  # 修正H本体: maxを取るだけ
    truth = df_v["opp_buried"].astype(int).values

    flag_old, sev_old = severity_flag(pending, counter_old_v, room_v)
    flag_new, sev_new = severity_flag(pending, counter_new_v, room_v)

    print(f"\n=== precision/recall (全体、n={len(df_v)}) — 修正前 vs 修正H後 ===")
    pr_old = precision_recall(flag_old, truth)
    pr_new = precision_recall(flag_new, truth)
    print(f"  修正前 (expected_fire_power単体) : {pr_old}")
    print(f"  修正H後(max(immediate,expected)): {pr_new}")

    n_counter_increased = int((counter_new_v > counter_old_v).sum())
    print(f"\n  counterが増加した行: {n_counter_increased}/{len(df_v)}"
          f" ({n_counter_increased / len(df_v):.1%}) (immediate_fire_powerがexpectedを上回った件数)")

    print("\n=== 校正曲線: 修正前 ===")
    calib_old = calibration_by_severity_bin(sev_old, truth)
    print(calib_old.to_string(index=False))
    calib_old.to_csv(OUT_DIR / "calibration_old.csv", index=False)

    print("\n=== 校正曲線: 修正H後 ===")
    calib_new = calibration_by_severity_bin(sev_new, truth)
    print(calib_new.to_string(index=False))
    calib_new.to_csv(OUT_DIR / "calibration_new.csv", index=False)

    summary = pd.DataFrame([{"版": "修正前", **pr_old}, {"版": "修正H後", **pr_new}])
    summary.to_csv(OUT_DIR / "precision_recall_summary.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
