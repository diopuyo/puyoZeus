"""修正H2 (既知ツモ列による反撃力の条件付け) 全域バックテスト (2026-08-04)。

18,057イベントで「受け切れ判定」(=severity>0、旧二値と同一の発動基準) を
予測とみなし、受け側の実窒息 (opp_buried) との precision/recall + 校正曲線を
修正H (immediate/expected の max) と修正H2 (既知ツモ2組を追加) で比較する。

既存資産の再利用のみ: reconstruct_event_board_pair (盤面復元)、
board_room、sim_expected_counter_ojama 列 (aug CSV既存)、immediate_fire_power
(軽量)、scripts.proto_net_threat_v2._predicted_counter_ojama_v2 (既知ツモ
評価、再実装しない)。opp側の next1_a/b・dnext_a/b は reconstruct_event_
board_pair と同じ「時刻最近傍」規則で別途取得する (関数の戻り値に含まれない
ため、同じ規則のみ再利用しboard自体の復元ロジックは再実装しない)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.indicators_v2 as iv
from scripts.compute_exchange_delta_winprob import (
    OJAMA_MAX_DROP_PER_TURN,
    _load_video_npz,
    reconstruct_event_board_pair,
)
from scripts.measure_exchange_effectiveness import estimate_available_hands
from scripts.proto_net_threat_v2 import _is_valid_next_pair, _predicted_counter_ojama_v2
from scripts.visualize_advantage_overlay import board_room
from src.chain import ChainSimulator

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/fix_h2_backtest_2026-08-04")
SEVERITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _opp_next_pair_at(cache, receiver_side: str, game_idx: int, t_sec: float):
    """reconstruct_event_board_pair と同じ「時刻最近傍」規則でopp側の
    next_pair/dnext_pairを取得する (盤面復元ロジック自体は再利用のみ)。
    """
    rec = cache.r2p if receiver_side == "2P" else cache.r1p
    mask = rec.game_idx == game_idx
    t_arr = rec.t_sec[mask]
    if len(t_arr) == 0:
        return None, None
    nearest = int(np.argmin(np.abs(t_arr - t_sec)))
    n1a, n1b = rec.next1_a[mask], rec.next1_b[mask]
    d1a, d1b = rec.dnext_a[mask], rec.dnext_b[mask]
    if len(n1a) == 0:
        return (-1, -1), (-1, -1)
    return (int(n1a[nearest]), int(n1b[nearest])), (int(d1a[nearest]), int(d1b[nearest]))


def compute_room_immediate_and_known(df: pd.DataFrame, sim: ChainSimulator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """行ごとに room / immediate_fire_power / 既知ツモ評価値 / 既知ツモ有効フラグ
    を npz から計算する (video単位でキャッシュ再利用)。10分おき進捗ログ
    (feedback_progress_notify_10min 準拠)。
    """
    import time
    n = len(df)
    room = np.full(n, np.nan)
    immediate = np.full(n, np.nan)
    known_queue = np.full(n, np.nan)
    next_valid = np.zeros(n, dtype=bool)
    t_start = time.time()
    done = 0
    for video_id, sub in df.groupby("video_id"):
        stem = video_id[len("video_"):] if video_id.startswith("video_") else video_id
        cache = _load_video_npz(stem, REGEN_NPZ_DIR)
        if cache is None:
            continue
        for idx, row in sub.iterrows():
            game_idx, t_sec, fire_side = int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"])
            pair = reconstruct_event_board_pair(cache, game_idx, t_sec, fire_side)
            if pair is None:
                continue
            _fire_board, opp_board = pair
            i = df.index.get_loc(idx)
            room[i] = board_room(opp_board)
            immediate[i] = iv.immediate_fire_power(opp_board).raw
            receiver_side = "2P" if fire_side == "1P" else "1P"
            next_pair, dnext_pair = _opp_next_pair_at(cache, receiver_side, game_idx, t_sec)
            if _is_valid_next_pair(next_pair):
                next_valid[i] = True
                k_hands = estimate_available_hands(int(round(float(row["approx_fire_chains"]))))
                known_queue[i], _ = _predicted_counter_ojama_v2(
                    opp_board, 0.0, k_hands, next_pair, dnext_pair, sim)
            done += 1
            if done % 1000 == 0:
                elapsed_min = (time.time() - t_start) / 60.0
                remaining_min = elapsed_min / done * (n - done)
                print(f"[PROGRESS] {done}/{n} 完了 (経過{elapsed_min:.1f}分、残り約{remaining_min:.1f}分)", flush=True)
    return room, immediate, known_queue, next_valid


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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {"n_pred_positive": int(tp + fp), "n_true_positive": int(tp + fn),
            "precision": precision, "recall": recall, "f1": f1}


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
    sim = ChainSimulator()

    room, immediate, known_queue, next_valid = compute_room_immediate_and_known(df, sim)
    n_ok = int(np.sum(~np.isnan(room)))
    print(f"盤面復元成功: {n_ok}/{len(df)} ({n_ok / len(df):.1%})")
    print(f"既知ツモ(next_pair)有効: {int(next_valid.sum())}/{len(df)} ({next_valid.mean():.1%})")

    valid = ~np.isnan(room)
    df_v = df.loc[valid].reset_index(drop=True)
    room_v, immediate_v, known_v = room[valid], immediate[valid], known_queue[valid]
    pending = df_v["net_ojama_after"].astype(float).values
    counter_h_v = np.maximum(df_v["sim_expected_counter_ojama"].astype(float).values, immediate_v)
    counter_h2_v = np.where(np.isnan(known_v), counter_h_v, np.maximum(counter_h_v, known_v))
    truth = df_v["opp_buried"].astype(int).values

    flag_h, sev_h = severity_flag(pending, counter_h_v, room_v)
    flag_h2, sev_h2 = severity_flag(pending, counter_h2_v, room_v)

    print(f"\n=== precision/recall/F1 (全体、n={len(df_v)}) — 修正H vs 修正H2 ===")
    pr_h = precision_recall(flag_h, truth)
    pr_h2 = precision_recall(flag_h2, truth)
    print(f"  修正H : {pr_h}")
    print(f"  修正H2: {pr_h2}")

    n_h2_helped = int(((counter_h2_v > counter_h_v)).sum())
    print(f"\n  既知ツモがcounterを押し上げた行: {n_h2_helped}/{len(df_v)} ({n_h2_helped / len(df_v):.1%})")

    print("\n=== 校正曲線: 修正H ===")
    calib_h = calibration_by_severity_bin(sev_h, truth)
    print(calib_h.to_string(index=False))
    calib_h.to_csv(OUT_DIR / "calibration_h.csv", index=False)

    print("\n=== 校正曲線: 修正H2 ===")
    calib_h2 = calibration_by_severity_bin(sev_h2, truth)
    print(calib_h2.to_string(index=False))
    calib_h2.to_csv(OUT_DIR / "calibration_h2.csv", index=False)

    summary = pd.DataFrame([{"版": "修正H", **pr_h}, {"版": "修正H2", **pr_h2}])
    summary.to_csv(OUT_DIR / "precision_recall_summary.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
