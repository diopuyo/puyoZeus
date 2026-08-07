"""user仮説の検証: 両者ともおじゃま4個以下のクリーン盤面に限定した色ぷよ量差 vs 実勝率。

前回の「おじゃま差中立 (|diff|<=3)」は差ベースで両者とも埋まったペアを含む。
今回は絶対条件 (両者<=4個) = 純粋な材料競争の局面のみ。
"""
import sys
sys.path.insert(0, ".")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "base", "scripts/_measure_puyo_lead_vs_winrate_2026-08-04.py")
base = importlib.util.module_from_spec(spec)
sys.modules["base"] = base
spec.loader.exec_module(base)

OJAMA_CLEAN_MAX = 4  # user指定: お互い4個以下

def main() -> None:
    df = base.load_labeled_csv(str(base.DEFAULT_LABELED_WIN_CSV))
    paired = base.pair_sides_for_win(df, base.PAIR_MAX_TDIFF_SEC)
    phase_metric = (paired["board_puyo_total_1p"].astype(float).values
                    + paired["board_puyo_total_2p"].astype(float).values)
    phase_labels, _, _ = base._assign_phase_by_puyo_tertile(phase_metric)
    paired = paired.copy()
    paired["phase"] = phase_labels
    paired["color_puyo_diff_raw"] = (
        paired["board_color_puyo_total_raw_1p"].astype(float)
        - paired["board_color_puyo_total_raw_2p"].astype(float))
    paired["ojama_count_diff_raw"] = (
        paired["board_ojama_count_raw_1p"].astype(float)
        - paired["board_ojama_count_raw_2p"].astype(float))
    clean = paired[
        (paired["board_ojama_count_raw_1p"].astype(float) <= OJAMA_CLEAN_MAX)
        & (paired["board_ojama_count_raw_2p"].astype(float) <= OJAMA_CLEAN_MAX)
    ].copy()
    clean["color_bucket"] = base._bucketize(clean["color_puyo_diff_raw"].values)
    print(f"クリーン盤面ペア (両者おじゃま<={OJAMA_CLEAN_MAX}): {len(clean)} / 全{len(paired)}")
    for phase in ["序", "中", "終"]:
        sub = clean[clean["phase"] == phase]
        print(f"\n=== {phase} (n={len(sub)}) ===")
        print(base._summarize_by_bucket(sub, "color_bucket",
                                        other_diff_col="ojama_count_diff_raw").to_string())

if __name__ == "__main__":
    main()

def main_no_inflight() -> None:
    """さらに発火隣接窓を除外: 両者とも直近の発火イベントから離れた純材料競争のみ。"""
    import pandas as pd
    ev = pd.read_csv("data/indicators_v2/exchange_labels_regen_synth79_2026-08-04.csv")
    df = base.load_labeled_csv(str(base.DEFAULT_LABELED_WIN_CSV))
    paired = base.pair_sides_for_win(df, base.PAIR_MAX_TDIFF_SEC)
    phase_metric = (paired["board_puyo_total_1p"].astype(float).values
                    + paired["board_puyo_total_2p"].astype(float).values)
    phase_labels, _, _ = base._assign_phase_by_puyo_tertile(phase_metric)
    paired = paired.copy()
    paired["phase"] = phase_labels
    paired["color_puyo_diff_raw"] = (
        paired["board_color_puyo_total_raw_1p"].astype(float)
        - paired["board_color_puyo_total_raw_2p"].astype(float))
    paired["ojama_count_diff_raw"] = (
        paired["board_ojama_count_raw_1p"].astype(float)
        - paired["board_ojama_count_raw_2p"].astype(float))
    clean = paired[
        (paired["board_ojama_count_raw_1p"].astype(float) <= OJAMA_CLEAN_MAX)
        & (paired["board_ojama_count_raw_2p"].astype(float) <= OJAMA_CLEAN_MAX)
    ].copy()
    # 発火隣接除外: 同一動画で |pair_t - event_t| < 12秒 の発火 (どちら側でも) があれば除外
    WINDOW_SEC = 12.0
    ev_by_video = {v: g["t_sec"].values for v, g in ev.groupby("video_id")}
    import numpy as np
    keep = np.ones(len(clean), dtype=bool)
    tcol = "t_sec_1p" if "t_sec_1p" in clean.columns else ("t_sec" if "t_sec" in clean.columns else None)
    assert tcol, f"時刻列が見つからない: {list(clean.columns)[:20]}"
    for i, (vid, t) in enumerate(zip(clean["video_id_1p"].values, clean[tcol].values)):
        evs = ev_by_video.get(vid)
        if evs is not None and len(evs) and np.min(np.abs(evs - float(t))) < WINDOW_SEC:
            keep[i] = False
    pure = clean[keep].copy()
    pure["color_bucket"] = base._bucketize(pure["color_puyo_diff_raw"].values)
    print(f"純材料競争ペア (クリーン盤面 かつ 発火±{WINDOW_SEC}s除外): {len(pure)} / クリーン{len(clean)}")
    for phase in ["序", "中", "終"]:
        sub = pure[pure["phase"] == phase]
        print(f"\n=== {phase} (n={len(sub)}) ===")
        print(base._summarize_by_bucket(sub, "color_bucket",
                                        other_diff_col="ojama_count_diff_raw").to_string())

