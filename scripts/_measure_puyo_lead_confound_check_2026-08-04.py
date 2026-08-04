"""J-1補足 (2026-08-04): おじゃま数差をほぼ中立(-3~+3)に絞った部分集合で、
色ぷよ量差バケット別の実勝率を再測定する (交絡分離: 色ぷよリード単独の
効果を、おじゃま押し付けられ由来の影響から分離するため)。
"""
from __future__ import annotations

from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV, PAIR_MAX_TDIFF_SEC, _assign_phase_by_puyo_tertile,
)
from scripts.model_indicator_win import load_labeled_csv, pair_sides_for_win

import importlib
_m = importlib.import_module("scripts._measure_puyo_lead_vs_winrate_2026-08-04")


def main() -> None:
    df = load_labeled_csv(str(DEFAULT_LABELED_WIN_CSV))
    paired = pair_sides_for_win(df, PAIR_MAX_TDIFF_SEC)
    phase_metric = (paired["board_puyo_total_1p"].astype(float).values
                    + paired["board_puyo_total_2p"].astype(float).values)
    phase_labels, _q_low, _q_high = _assign_phase_by_puyo_tertile(phase_metric)
    paired = paired.copy()
    paired["phase"] = phase_labels
    paired["color_puyo_diff_raw"] = (
        paired["board_color_puyo_total_raw_1p"].astype(float)
        - paired["board_color_puyo_total_raw_2p"].astype(float)
    )
    paired["ojama_count_diff_raw"] = (
        paired["board_ojama_count_raw_1p"].astype(float)
        - paired["board_ojama_count_raw_2p"].astype(float)
    )
    paired["color_bucket"] = _m._bucketize(paired["color_puyo_diff_raw"].values)

    # おじゃま数差がほぼ中立(-3~+3)な行だけに絞る (交絡分離)。
    neutral_ojama = paired.loc[paired["ojama_count_diff_raw"].abs() <= 3.0]
    for phase in ("序", "中", "終"):
        sub = neutral_ojama.loc[neutral_ojama["phase"] == phase]
        print(f"\n=== 位相={phase} おじゃま数差ほぼ中立(|diff|<=3)のみ (n={len(sub)}) ===")
        print(_m._summarize_by_bucket(sub, "color_bucket", other_diff_col="ojama_count_diff_raw")
              .to_string(index=False))


if __name__ == "__main__":
    main()
