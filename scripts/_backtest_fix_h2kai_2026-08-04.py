"""修正H2改 (既知ツモの用途を「構え完成1手前」に限定) 全域バックテスト (2026-08-04)。

18,057イベントで「受け切れ判定」(=severity>0、旧二値と同一の発動基準) を
予測とみなし、受け側の実窒息 (opp_buried) との precision/recall/F1 + 校正曲線を
修正H (immediate/expected の max) と修正H2改 (既知ツモ単独配置の完成値を追加)
で比較する。既存資産の再利用のみ: reconstruct_event_board_pair、board_room、
sim_expected_counter_ojama列 (aug CSV既存)、immediate_fire_power、
scripts.compute_exchange_delta_winprob._known_pair_completion_value (本番と
同一関数、再実装しない)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import src.indicators_v2 as iv
from scripts.compute_exchange_delta_winprob import (
    OJAMA_MAX_DROP_PER_TURN,
    _known_pair_completion_value,
    _load_video_npz,
    reconstruct_event_board_pair,
)
from scripts.visualize_advantage_overlay import board_room

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OUT_DIR = Path("data/verify/fix_h2kai_backtest_2026-08-04")
SEVERITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _opp_next_pair_at(cache, receiver_side: str, game_idx: int, t_sec: float):
    """reconstruct_event_board_pair と同じ「時刻最近傍」規則でopp側の
    next_pair/dnext_pairを取得する (盤面復元ロジック自体は再利用のみ)。
    """
    rec = cache.r2p if receiver_side == "2P" else cache.r1p
    mask = rec.game_idx == game_idx
    t_arr = rec.t_sec[mask]
    if len(t_arr) == 0:
        return (-1, -1), (-1, -1)
    nearest = int(np.argmin(np.abs(t_arr - t_sec)))
    n1a, n1b = rec.next1_a[mask], rec.next1_b[mask]
    d1a, d1b = rec.dnext_a[mask], rec.dnext_b[mask]
    if len(n1a) == 0:
        return (-1, -1), (-1, -1)
    return (int(n1a[nearest]), int(n1b[nearest])), (int(d1a[nearest]), int(d1b[nearest]))


def compute_room_immediate_and_completion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """行ごとに room / immediate_fire_power / 構え完成値 / 完成値>0フラグ
    を npz から計算する (video単位でキャッシュ再利用)。10分おき進捗ログ。
    """
    n = len(df)
    room = np.full(n, np.nan)
    immediate = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    completion_positive = np.zeros(n, dtype=bool)
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
            completion[i] = _known_pair_completion_value(opp_board, next_pair, dnext_pair)
            completion_positive[i] = completion[i] > 0.0
            done += 1
            if done % 1000 == 0:
                elapsed_min = (time.time() - t_start) / 60.0
                remaining_min = elapsed_min / done * (n - done)
                print(f"[PROGRESS] {done}/{n} 完了 (経過{elapsed_min:.1f}分、残り約{remaining_min:.1f}分)", flush=True)
    return room, immediate, completion, completion_positive


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


SAMPLE_FRAC = 0.30  # 実行時間対策のサンプリング (video単位の層化、全域=1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_full = pd.read_csv(AUG_CSV)
    if SAMPLE_FRAC < 1.0:
        rng = np.random.RandomState(42)
        parts = [g.sample(frac=SAMPLE_FRAC, random_state=rng) for _, g in df_full.groupby("video_id")]
        df = pd.concat(parts).sort_index().reset_index(drop=True)
        print(f"=== 入力: 全{len(df_full)}行のうちvideo層化サンプリング{SAMPLE_FRAC:.0%}"
              f" = {len(df)}行 (実行時間対策、動画分布は維持) ===")
    else:
        df = df_full
        print(f"=== 入力: {len(df)}行 (全域) ===")

    room, immediate, completion, completion_positive = compute_room_immediate_and_completion(df)
    n_ok = int(np.sum(~np.isnan(room)))
    print(f"盤面復元成功: {n_ok}/{len(df)} ({n_ok / len(df):.1%})")
    print(f"構え完成値>0の行: {int(completion_positive.sum())}/{len(df)} ({completion_positive.mean():.1%})")

    valid = ~np.isnan(room)
    df_v = df.loc[valid].reset_index(drop=True)
    room_v, immediate_v, completion_v = room[valid], immediate[valid], completion[valid]
    pending = df_v["net_ojama_after"].astype(float).values
    counter_h_v = np.maximum(df_v["sim_expected_counter_ojama"].astype(float).values, immediate_v)
    counter_h2kai_v = np.maximum(counter_h_v, completion_v)

    # 2026-08-04 user決定: 理論値(受け切れ判定)チャネルの正解ラベルは
    # 「回避不能死」(=opp_buried かつ returned==0、既存ラベルの組み合わせ
    # のみ・新規計算なし) で再定義する。「死んだ≠決まっていた」の歪みを
    # 除くため、受け手が実際に発火して応戦した末に負けたケースは正解から
    # 除外する (H2改の検収はこちらを主指標とする)。
    truth_buried = df_v["opp_buried"].astype(int).values
    truth_unavoidable = (
        (df_v["opp_buried"].astype(int).values == 1) & (df_v["returned"].astype(int).values == 0)
    ).astype(int)

    flag_h, sev_h = severity_flag(pending, counter_h_v, room_v)
    flag_h2kai, sev_h2kai = severity_flag(pending, counter_h2kai_v, room_v)

    print(f"\n=== [参考] precision/recall/F1 (旧ラベル=opp_buried、n={len(df_v)}) ===")
    pr_h_old = precision_recall(flag_h, truth_buried)
    pr_h2kai_old = precision_recall(flag_h2kai, truth_buried)
    print(f"  修正H  : {pr_h_old}")
    print(f"  修正H2改: {pr_h2kai_old}")

    print(f"\n=== [主指標] precision/recall/F1 (新ラベル=回避不能死, n={len(df_v)}) ===")
    pr_h = precision_recall(flag_h, truth_unavoidable)
    pr_h2kai = precision_recall(flag_h2kai, truth_unavoidable)
    n_unavoidable = int(truth_unavoidable.sum())
    print(f"  回避不能死の件数: {n_unavoidable}/{len(df_v)} ({n_unavoidable / len(df_v):.1%})"
          f" (旧opp_buried={int(truth_buried.sum())}件からreturned=1を除外)")
    print(f"  修正H  : {pr_h}")
    print(f"  修正H2改: {pr_h2kai}")
    print(f"\n  [検収3] precision>=H({pr_h['precision']:.4f})? {pr_h2kai['precision'] >= pr_h['precision']}"
          f" / recall>=H({pr_h['recall']:.4f})? {pr_h2kai['recall'] >= pr_h['recall']}")

    n_helped = int((counter_h2kai_v > counter_h_v).sum())
    print(f"\n  構え完成値がcounterを押し上げた行: {n_helped}/{len(df_v)} ({n_helped / len(df_v):.1%})")

    print("\n=== 校正曲線: 修正H (回避不能死ラベル) ===")
    calib_h = calibration_by_severity_bin(sev_h, truth_unavoidable)
    print(calib_h.to_string(index=False))
    calib_h.to_csv(OUT_DIR / "calibration_h_unavoidable.csv", index=False)

    print("\n=== 校正曲線: 修正H2改 (回避不能死ラベル) ===")
    calib_h2kai = calibration_by_severity_bin(sev_h2kai, truth_unavoidable)
    print(calib_h2kai.to_string(index=False))
    calib_h2kai.to_csv(OUT_DIR / "calibration_h2kai_unavoidable.csv", index=False)

    summary = pd.DataFrame([{"版": "修正H", **pr_h}, {"版": "修正H2改", **pr_h2kai}])
    summary.to_csv(OUT_DIR / "precision_recall_summary_unavoidable.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
