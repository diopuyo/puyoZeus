"""#24 Step0 no_baseline (118件) の切り分け診断 (2026-07-29)。

目的:
    no_baseline (発火時点で相手側にその game_idx の STABLE 観測が1件も無い)
    が「試合境界の同期ズレバグ」由来か「試合開始直後の実態」かを切り分ける。

⚠️ 認識の再実行は一切行わない。既存 CSV
   (data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv) と
   既存 npz (data/indicators_v2/boards_lean_fixed_regen_2026-07-28/) の
   読み取り集計のみで完結する。scripts/measure_exchange_dynamics.py・
   scripts/measure_ojama_landing_delay.py は import のみで一切変更しない。

使い方:
    nice -n 19 venv/bin/python -m scripts._diag_no_baseline_2026-07-29
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from scripts.measure_exchange_dynamics import TIER_MAP  # noqa: E402
from scripts.measure_ojama_landing_delay import _load_npz  # noqa: E402

# 入力 (既存資産、無改変)
INPUT_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_landing_delay_regen_2026-07-28.csv"
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"

# 試合開始直後とみなす秒数閾値 (userタスクの「数秒以内」の目安、診断用)
EARLY_GAME_SEC_THRESHOLD: float = 5.0

# 同期ズレとみなす境界時刻差の秒数閾値 (userタスクの「食い違い」の目安、診断用)
BOUNDARY_DESYNC_SEC_THRESHOLD: float = 5.0


def _game_boundary_table(npz_path: Path) -> pd.DataFrame:
    """1動画分、game_idx ごとの 1P/2P 境界時刻 (最小/最大 t_sec) 一覧を返す。"""
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return pd.DataFrame()
    rows: list[dict] = []
    all_gids = sorted(set(by_side["1P"].game_idx.tolist()) | set(by_side["2P"].game_idx.tolist()))
    for gid in all_gids:
        row: dict = {"video_stem": npz_path.stem, "game_idx": gid}
        for side in ("1P", "2P"):
            rec = by_side[side]
            mask = rec.game_idx == gid
            if mask.any():
                t = rec.t_sec[mask]
                row[f"{side}_t_min"] = float(t.min())
                row[f"{side}_t_max"] = float(t.max())
                row[f"{side}_n"] = int(mask.sum())
            else:
                row[f"{side}_t_min"] = float("nan")
                row[f"{side}_t_max"] = float("nan")
                row[f"{side}_n"] = 0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["start_diff_sec"] = (df["1P_t_min"] - df["2P_t_min"]).abs()
    return df


def _annotate_no_baseline(df_nb: pd.DataFrame, boundary_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """no_baseline 行に、game_start_t / 自側開始からの経過秒 / 相手側開始との差を付与する。

    game_start_t の定義は _process_video (scripts/measure_exchange_dynamics.py:664)
    と同一 (両側の game_idx 内最小 t_sec のさらに min) に合わせる。
    """
    out_rows: list[dict] = []
    for _, row in df_nb.iterrows():
        stem = row["video_stem"]
        bt = boundary_tables.get(stem)
        rec = {**row.to_dict()}
        if bt is None or bt.empty:
            rec["elapsed_since_game_start_sec"] = float("nan")
            rec["own_side_game_first_t"] = float("nan")
            rec["opp_side_game_first_t"] = float("nan")
            rec["opp_first_minus_t_fire_sec"] = float("nan")
            out_rows.append(rec)
            continue
        gmatch = bt[bt["game_idx"] == row["game_idx"]]
        if gmatch.empty:
            rec["elapsed_since_game_start_sec"] = float("nan")
            rec["own_side_game_first_t"] = float("nan")
            rec["opp_side_game_first_t"] = float("nan")
            rec["opp_first_minus_t_fire_sec"] = float("nan")
            out_rows.append(rec)
            continue
        g = gmatch.iloc[0]
        own_side = row["fire_side"]
        opp_side = "2P" if own_side == "1P" else "1P"
        game_start_t = min(g["1P_t_min"], g["2P_t_min"])
        rec["elapsed_since_game_start_sec"] = float(row["t_fire"]) - game_start_t
        rec["own_side_game_first_t"] = g[f"{own_side}_t_min"]
        rec["opp_side_game_first_t"] = g[f"{opp_side}_t_min"]
        rec["opp_first_minus_t_fire_sec"] = g[f"{opp_side}_t_min"] - float(row["t_fire"])
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(INPUT_CSV)
    df_nb = df[df["detection_status"] == "no_baseline"].copy()
    print(f"[INFO] 全体 n={len(df)} / no_baseline n={len(df_nb)}")

    stems = sorted(df_nb["video_stem"].unique())
    boundary_tables: dict[str, pd.DataFrame] = {}
    for stem in sorted(TIER_MAP):
        npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
        if not npz_path.exists():
            continue
        boundary_tables[stem] = _game_boundary_table(npz_path)

    ann = _annotate_no_baseline(df_nb, boundary_tables)

    # --- Q1: 試合開始からの経過秒分布 ---
    print("\n=== Q1: no_baseline の t_fire が試合開始から何秒後か ===")
    elapsed = ann["elapsed_since_game_start_sec"].dropna()
    print(f"n={len(elapsed)}")
    print(f"  中央値 = {elapsed.median():.2f} 秒")
    print(f"  25%tile = {elapsed.quantile(0.25):.2f} 秒")
    print(f"  75%tile = {elapsed.quantile(0.75):.2f} 秒")
    print(f"  最大値 = {elapsed.max():.2f} 秒")
    print(f"  最小値 = {elapsed.min():.2f} 秒")
    n_early = int((elapsed <= EARLY_GAME_SEC_THRESHOLD).sum())
    print(f"  試合開始 {EARLY_GAME_SEC_THRESHOLD}秒以内: {n_early}/{len(elapsed)} "
          f"({n_early / len(elapsed) * 100:.1f}%)")
    print("  [経過秒 ヒストグラム(粗いビン)]")
    bins = [0, 5, 10, 30, 60, 120, 300, np.inf]
    labels = ["0-5s", "5-10s", "10-30s", "30-60s", "60-120s", "120-300s", "300s+"]
    cats = pd.cut(elapsed, bins=bins, labels=labels)
    print(cats.value_counts().sort_index().to_string())

    # --- Q2: 1P/2P 偏り ---
    print("\n=== Q2: fire_side (攻撃側) 別の no_baseline 件数 ===")
    print(df_nb["fire_side"].value_counts().to_string())
    print("\n[動画別 x fire_side 内訳]")
    print(df_nb.groupby(["video_stem", "fire_side"]).size().unstack(fill_value=0).to_string())

    # --- 動画別 no_baseline 件数 (c80優先確認の根拠) ---
    print("\n=== 動画別 no_baseline 件数 (全発火数との比) ===")
    n_by_video = df.groupby("video_stem").size().rename("n_fire_total")
    nb_by_video = df_nb.groupby("video_stem").size().rename("n_no_baseline")
    summary = pd.concat([n_by_video, nb_by_video], axis=1).fillna(0)
    summary["n_no_baseline"] = summary["n_no_baseline"].astype(int)
    summary["pct"] = summary["n_no_baseline"] / summary["n_fire_total"] * 100.0
    print(summary.sort_values("n_no_baseline", ascending=False).to_string())

    # --- Q3: game_idx 境界の同期ズレ直接確認 (全動画) ---
    print("\n=== Q3: 動画別 game_idx 境界 (1P_t_min vs 2P_t_min) の食い違い ===")
    all_bt = pd.concat(boundary_tables.values(), ignore_index=True) if boundary_tables else pd.DataFrame()
    if not all_bt.empty:
        both_present = all_bt[(all_bt["1P_n"] > 0) & (all_bt["2P_n"] > 0)]
        desync = both_present[both_present["start_diff_sec"] > BOUNDARY_DESYNC_SEC_THRESHOLD]
        print(f"両側にフレームがある game_idx 総数: {len(both_present)}")
        print(f"start_diff_sec > {BOUNDARY_DESYNC_SEC_THRESHOLD}秒 の件数: {len(desync)}")
        if not desync.empty:
            print(desync[["video_stem", "game_idx", "1P_t_min", "2P_t_min", "start_diff_sec"]]
                  .sort_values("start_diff_sec", ascending=False).head(20).to_string(index=False))
        else:
            print("  (該当なし: 全 game_idx で両側の開始時刻は概ね一致)")

    # --- c80 詳細 (最多動画) ---
    print("\n=== c80 詳細: game_idx 境界一覧 (1P vs 2P) ===")
    if "c80" in boundary_tables:
        bt80 = boundary_tables["c80"]
        print(bt80[["game_idx", "1P_t_min", "1P_t_max", "1P_n",
                     "2P_t_min", "2P_t_max", "2P_n", "start_diff_sec"]].to_string(index=False))

    print("\n=== c80 の no_baseline 発火イベント詳細 ===")
    ann80 = ann[ann["video_stem"] == "c80"]
    if not ann80.empty:
        cols = ["game_idx", "fire_side", "t_fire", "elapsed_since_game_start_sec",
                "own_side_game_first_t", "opp_side_game_first_t", "opp_first_minus_t_fire_sec"]
        print(ann80[cols].sort_values("game_idx").to_string(index=False))
    else:
        print("  (c80 の no_baseline 行なし)")

    # --- opp_first_minus_t_fire_sec の分布 (バグ判定の核心) ---
    print("\n=== 核心指標: opp_side の game内最初のSTABLE時刻 - t_fire (秒) ===")
    print("  正値が大きいほど「相手はこのgame_idxで長時間観測皆無」= 実態(試合序盤)寄り")
    print("  正値が小さい(数秒)なら「相手の観測開始が単に少し遅れた」= 認識起動ラグ寄り")
    diff = ann["opp_first_minus_t_fire_sec"].dropna()
    print(f"n={len(diff)}")
    print(f"  中央値 = {diff.median():.2f} 秒")
    print(f"  75%tile = {diff.quantile(0.75):.2f} 秒")
    print(f"  最大値 = {diff.max():.2f} 秒")
    print(f"  最小値 = {diff.min():.2f} 秒")

    # --- 核心検証: game_idx を無視して、t_fire 近傍に opp の実データが
    # 存在するかを直接確認する (バグ由来か実態かを断定するための唯一の直接証拠)。
    print("\n=== 核心検証: game_idx を無視し、opp 側に t_fire 近傍(±10秒)のフレームが実在するか ===")
    npz_cache: dict[str, dict[str, object]] = {}
    n_recoverable = 0
    n_checked = 0
    recoverable_rows: list[dict] = []
    for _, row in df_nb.iterrows():
        stem = row["video_stem"]
        if stem not in npz_cache:
            npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
            records = _load_npz(npz_path)
            npz_cache[stem] = {r.side: r for r in records}
        by_side = npz_cache[stem]
        own_side = row["fire_side"]
        opp_side = "2P" if own_side == "1P" else "1P"
        if opp_side not in by_side:
            continue
        opp_rec = by_side[opp_side]
        t_fire = float(row["t_fire"])
        near_mask = np.abs(opp_rec.t_sec - t_fire) <= 10.0
        n_checked += 1
        if near_mask.any():
            n_recoverable += 1
            near_gidx = sorted(set(opp_rec.game_idx[near_mask].tolist()))
            recoverable_rows.append({
                "video_stem": stem, "game_idx_fire_side": row["game_idx"],
                "fire_side": own_side, "t_fire": t_fire,
                "opp_actual_game_idx_near_t_fire": near_gidx,
                "n_opp_frames_near": int(near_mask.sum()),
            })
    print(f"確認対象 no_baseline n={n_checked}")
    print(f"opp側に t_fire±10秒の実フレームが存在する (=game_idxラベルさえ正しければ着弾判定可能だった) 件数: "
          f"{n_recoverable}/{n_checked} ({n_recoverable / n_checked * 100:.1f}%)")
    if recoverable_rows:
        print("\n[該当例 (先頭10件): fire側の game_idx とは異なる game_idx で opp データが実在]")
        rec_df = pd.DataFrame(recoverable_rows)
        mismatch = rec_df[rec_df.apply(
            lambda r: r["game_idx_fire_side"] not in r["opp_actual_game_idx_near_t_fire"], axis=1,
        )]
        print(f"  うち opp実データの game_idx が fire側と完全不一致: {len(mismatch)}/{len(rec_df)}")
        print(mismatch.head(10).to_string(index=False))

    out_path = PROJ_ROOT / "data" / "indicators_v2" / "_diag_no_baseline_annotated_2026-07-29.csv"
    ann.to_csv(out_path, index=False)
    print(f"\n[DONE] 注釈付き明細を {out_path} に保存しました")


if __name__ == "__main__":
    main()
