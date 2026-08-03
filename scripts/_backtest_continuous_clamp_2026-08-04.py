"""連続クランプの全域バックテスト (2026-08-04 user恒久指示: 過学習ガード)。

18,057イベント (data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv)
で「受け切れ判定」を予測とみなし、受け側の実窒息 (opp_buried、既存ラベル
再利用) との precision/recall + 校正曲線 (severityビン別の実窒息率) を
二値版(旧)と連続版(severity>0.5)で比較する。

realizable_counter は aug CSV の sim_expected_counter_ojama 列をそのまま
使う (estimate_expected_net_damage 内部で _realizable_counter_ojama と
同一の k_hands=estimate_available_hands(approx_fire_chains) →
expected_fire_power(opp_board, k_levels=(k_hands,)) を計算済みのため、
再計算しない・既存資産再利用)。room のみ npz から盤面復元して計算する
(既存 reconstruct_event_board_pair を再利用、再実装しない)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_exchange_delta_winprob import (
    OJAMA_MAX_DROP_PER_TURN,
    _load_video_npz,
    reconstruct_event_board_pair,
)
from scripts.visualize_advantage_overlay import board_room

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/continuous_clamp_backtest_2026-08-04")
SEVERITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def compute_room_for_all_rows(df: pd.DataFrame) -> np.ndarray:
    """行ごとに受け手(opp)盤面の room を npz から復元する (video単位でキャッシュ再利用)。

    盤面復元に失敗した行 (突合失敗、主に合成終局イベント) はNaNにする
    (silent dropしない、件数は呼び出し元でログする)。
    """
    room = np.full(len(df), np.nan)
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
            room[df.index.get_loc(idx)] = board_room(opp_board)
    return room


def compute_severity_and_binary(pending: np.ndarray, counter: np.ndarray, room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """raw_excess = pending-counter-room から (二値flag, 連続severity) を計算する。"""
    raw_excess = pending - counter - room
    binary_flag = (raw_excess > 0.0).astype(float)
    severity = np.clip(raw_excess / OJAMA_MAX_DROP_PER_TURN, 0.0, 1.0)
    threatened = pending > 0.0
    binary_flag = np.where(threatened, binary_flag, 0.0)
    severity = np.where(threatened, severity, 0.0)
    return binary_flag, severity


def precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = float(((pred == 1) & (truth == 1)).sum())
    fp = float(((pred == 1) & (truth == 0)).sum())
    fn = float(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"n_pred_positive": int(tp + fp), "n_true_positive": int(tp + fn),
            "precision": precision, "recall": recall}


def calibration_by_severity_bin(severity: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """severityビン別の実窒息率 (校正曲線データ)。"""
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

    room = compute_room_for_all_rows(df)
    n_room_ok = int(np.sum(~np.isnan(room)))
    print(f"room 復元成功: {n_room_ok}/{len(df)} ({n_room_ok / len(df):.1%})")

    valid = ~np.isnan(room)
    df_valid = df.loc[valid].reset_index(drop=True)
    room_valid = room[valid]
    pending = df_valid["net_ojama_after"].astype(float).values
    counter = df_valid["sim_expected_counter_ojama"].astype(float).values
    truth = df_valid["opp_buried"].astype(int).values

    binary_flag, severity = compute_severity_and_binary(pending, counter, room_valid)
    # 2026-08-04 main最終判断(案a): 「発動判定」はseverity>0 (=旧二値の
    # raw_excess>0と完全同一、ヒステリシス込み)。severityは表示ブレンド量
    # 専用であり、判定基準としては使わない (severity>0.5指定は撤回済み)。
    severity_flag_00 = (severity > 0.0).astype(float)  # 公式の発動判定 (旧と同一)

    print(f"\n=== precision/recall (全体、n={len(df_valid)}) — 公式の発動判定=severity>0 ===")
    pr_binary = precision_recall(binary_flag, truth)
    pr_official = precision_recall(severity_flag_00, truth)
    print(f"  二値版(旧)                          : {pr_binary}")
    print(f"  連続版(severity>0、=案a公式の発動判定): {pr_official}")
    assert pr_binary == pr_official, "発動判定は旧二値と完全一致するはず (案aの前提)"
    print("  [確認OK] 発動判定のprecision/recallは旧二値と完全一致 (定義上無悪化)")

    n_synth_valid = int(df_valid["is_synthetic_terminal_event"].sum())
    print(f"\n  (参考: 上記のうち合成行={n_synth_valid}件)")

    print("\n=== 校正曲線: 二値版 (severity=0/1相当) ===")
    calib_binary = calibration_by_severity_bin(binary_flag, truth)
    print(calib_binary.to_string(index=False))
    calib_binary.to_csv(OUT_DIR / "calibration_binary.csv", index=False)

    print("\n=== 校正曲線: 連続版 (severityビン別) ===")
    calib_cont = calibration_by_severity_bin(severity, truth)
    print(calib_cont.to_string(index=False))
    calib_cont.to_csv(OUT_DIR / "calibration_continuous.csv", index=False)

    summary = pd.DataFrame([
        {"版": "二値(旧)", **pr_binary},
        {"版": "連続版(severity>0、案a公式)", **pr_official},
    ])
    summary.to_csv(OUT_DIR / "precision_recall_summary.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
