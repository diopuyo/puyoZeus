"""修正C/D 検収: 5検収シーンの勝率タイムライン全体(0.2秒刻み)を抽出しPNG化する。

render_delta_winprob_demo.py と同じ compute_display_state を直接呼ぶため、
実際のレンダ結果と数値的に完全一致する (cv2動画デコードは行わない軽量版)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.chain import ChainSimulator
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _build_stable_timeline,
    _load_video_npz,
    build_chain_in_progress_windows,
    build_event_activity_windows,
    train_winprob_models,
)
from scripts.render_delta_winprob_demo import build_fire_event_views, compute_display_state

NPZ_DIR = Path("data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03")
DELTA_CSV = Path("data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv")
OUT_DIR = Path("data/verify/fixCD_scene_verify_2026-08-03")
GRID_STEP_SEC = 0.2

# 検収5シーン: (game_idx, 窓開始, 窓終了, 指摘時刻一覧, 期待する物語)
# 2026-08-03 再指摘反映: match_02 のアンカーは user原文基準で2988s (2996.5は
# main誤訳のため撤回)。match_05 は51秒地点(=3163s)と3168sを明示。
SCENES = [
    (1, 2925.0, 2948.0, [2929.0, 2933.0, 2939.17, 2941.47],
     "2933以降1P決定的優勢が終局まで持続(2P側への谷なし)"),
    (2, 2980.0, 3010.0, [2984.9, 2988.0, 2992.93, 3006.9],
     "2988でのべ1P100%平坦が消滅し、2Pのぷよ量蓄積を反映した漸進的変化になる"),
    (4, 3090.0, 3110.0, [3093.2, 3097.3, 3104.8],
     "3097.3で大振れしない"),
    (5, 3150.0, 3175.0, [3154.1, 3157.5, 3157.8, 3158.8, 3163.0, 3168.0],
     "51秒(3163)/3168 で1P有利方向 (少なくとも2P優勢75%ではない)"),
]


def main() -> None:
    meiryo_path = "/mnt/c/Windows/Fonts/meiryo.ttc"
    if Path(meiryo_path).exists():
        font_manager.fontManager.addfont(meiryo_path)
        plt.rcParams["font.family"] = "Meiryo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 勝率モデル学習 ===")
    models = train_winprob_models(DEFAULT_LABELED_WIN_CSV)
    cache = _load_video_npz("olRyxDGacbg", NPZ_DIR)
    delta_df = pd.read_csv(DELTA_CSV)
    sim = ChainSimulator()

    for game_idx, t0, t1, marks, narrative in SCENES:
        events_df = delta_df.loc[(delta_df["game_idx"] == game_idx) & (~delta_df["match_failed"])].copy()
        windows = build_chain_in_progress_windows(events_df, cache, sim)
        activity_windows = build_event_activity_windows(events_df, cache)
        timeline_df = _build_stable_timeline(cache, game_idx, models, sim, windows, activity_windows)
        timeline_t = timeline_df["t_sec"].values.astype(float)
        timeline_v = timeline_df["winprob_1p"].values.astype(float)
        timeline_uncertain = (timeline_df["is_uncertain"].values.astype(bool)
                               if "is_uncertain" in timeline_df.columns else None)
        events = build_fire_event_views(events_df, cache)

        grid = np.arange(t0, t1, GRID_STEP_SEC)
        vals, uncertain_flags = [], []
        for t in grid:
            state = compute_display_state(events, timeline_t, timeline_v, t, timeline_uncertain)
            vals.append(np.nan if state.waiting else state.winprob_1p)
            uncertain_flags.append(state.uncertain_frozen)
        vals = np.array(vals)
        uncertain_flags = np.array(uncertain_flags)

        print(f"\n=== game_idx={game_idx} 窓=({t0:.1f},{t1:.1f}) 期待物語: {narrative} ===")
        print(f"  カバーする時間帯: {grid[0]:.1f}s 〜 {grid[-1]:.1f}s ({len(grid)}点, {GRID_STEP_SEC}s刻み)")
        valid = ~np.isnan(vals)
        print(f"  1P視点勝率: min={np.nanmin(vals):.1f} max={np.nanmax(vals):.1f}"
              f" 最終値={vals[valid][-1]:.1f} (waiting除外{int((~valid).sum())}点)"
              f" (判定保留{int(uncertain_flags.sum())}点)")
        for m in marks:
            idx = int(np.argmin(np.abs(grid - m)))
            flag = " [判定保留]" if uncertain_flags[idx] else ""
            print(f"  t={m:.2f}s 付近(grid={grid[idx]:.2f}): winprob_1p={vals[idx]:.1f}{flag}")

        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(grid, vals, color="steelblue", linewidth=1.2)
        ax.axhline(50.0, color="gray", linewidth=0.7, linestyle=":")
        if uncertain_flags.any():
            ax.fill_between(grid, -5, 105, where=uncertain_flags, color="grey", alpha=0.25,
                             step="post", label="判定保留(相手データ凍結中)")
            ax.legend(loc="upper right", fontsize=8)
        for m in marks:
            ax.axvline(m, color="orange", alpha=0.5, linewidth=1.0)
        for ev in events:
            col = "tab:blue" if ev.fire_side == "1P" else "tab:red"
            ax.axvline(ev.ignition_sec, color=col, alpha=0.3, linewidth=1.5, linestyle="--")
        ax.set_ylim(-5, 105)
        ax.set_xlabel("動画内 経過秒 (t_sec)")
        ax.set_ylabel("1P視点 勝率 (%)")
        ax.set_title(f"game_idx={game_idx}  期待物語: {narrative}")
        fig.tight_layout()
        out_path = OUT_DIR / f"scene_game{game_idx}_{int(t0)}_{int(t1)}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  PNG保存: {out_path}")


if __name__ == "__main__":
    main()
