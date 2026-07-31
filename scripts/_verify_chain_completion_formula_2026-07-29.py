"""軽量検証: 掛け算表示ベース連鎖完了時刻 (chain_completion_from_formula) の
妥当性チェック (2026-07-29)。

読み取り専用の診断スクリプト。src/、scripts/measure_exchange_dynamics.py、
scripts/measure_ojama_landing_delay.py、scripts/_diag_chain_anim_duration_multi.py
は import のみで一切変更しない (既存資産の再利用に徹する、追加収集14並列と
競合しないよう動画I/O・認識の再実行は一切行わない)。

検証内容:
    1. 母集団実測 (data/verify/recognition_diag_chain_anim_duration_multi/
       events_raw.csv、23動画418イベント) の chain_count ビン別に、
       新方式 (CHAIN_ANIM_PER_STEP_SEC=0.4 × chain_count、
       formula_appear_sec の代わりに t_chain_start を近似値として使用) が
       視覚実測 (visual_duration_sec、盤面が完全に静止するまで) の
       レンジにどう収まるかを比較する。
       ⚠️ visual_duration_sec は「盤面settle」(おじゃま落下+次ツモ出現を
       含む) であり、掛け算表示終了時刻そのものの実測ではない
       (docs/report 側で解釈を明記)。

    2. user指定の外れ値6件 (c62/c44/c59/c21/c54、うち c62 は2件report済み
       重複整理のため実質5動画) について、新方式の完了時刻 (t_chain_start
       基準) が視覚実測の同一 chain_count ビンの [min, max] に収まるかを
       個別に確認する。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/_verify_chain_completion_formula_2026-07-29.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.indicators_v2 import CHAIN_ANIM_PER_STEP_SEC, chain_completion_from_formula  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    CHAIN_BIN_CAP, NPZ_DIR, TIER_MAP, _process_video, _subset,
)

# 外れ値6件 (user提示、scripts/_diag_tfire_reliability_2026-07-29.py と同一)
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
TARGET_CASES: list[tuple[str, str, int, float]] = [
    ("c62", "2P", 6, 700.4),
    ("c44", "2P", 2, 257.8),
    ("c59", "1P", 1, 261.2),
    ("c21", "2P", 2, 279.8),
    ("c54", "1P", 1, 252.6),
]


def _population_bin_comparison() -> pd.DataFrame:
    """母集団 (23動画418イベント) の chain_count ビン別に新方式を比較する。"""
    csv_path = PROJ_ROOT / "data/verify/recognition_diag_chain_anim_duration_multi/events_raw.csv"
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    ok["chain_bin"] = ok["chain_count"].clip(upper=CHAIN_BIN_CAP)
    # 新方式: formula_appear_sec の代わりに t_chain_start を使う近似
    # (実データに formula_appear_sec が存在しないため、t_chain_start を
    # 「下限に近い近似値」として明示的に使用する。過小評価注意)。
    ok["new_end_rel"] = ok.apply(
        lambda r: chain_completion_from_formula(0.0, r["chain_count"]), axis=1,
    )
    g = ok.groupby("chain_bin").agg(
        n=("visual_duration_sec", "count"),
        visual_median=("visual_duration_sec", "median"),
        visual_min=("visual_duration_sec", "min"),
        visual_max=("visual_duration_sec", "max"),
        pipeline_median=("pipeline_duration_sec", "median"),
        new_formula_rel=("new_end_rel", "median"),
    )
    g["new_within_visual_range"] = (g["new_formula_rel"] >= g["visual_min"]) & (
        g["new_formula_rel"] <= g["visual_max"]
    )
    return g


def _outlier_case_check() -> pd.DataFrame:
    """外れ値6件を新方式で再計算し、母集団レンジと比較する。"""
    bins_csv = PROJ_ROOT / "data/verify/recognition_diag_chain_anim_duration_multi/chain_count_bins.csv"
    bins = pd.read_csv(bins_csv).set_index("chain_bin")

    sim = ChainSimulator()
    rows = []
    for stem, side, game_idx, t_cs_approx in TARGET_CASES:
        npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
        if not npz_path.exists():
            npz_path = NPZ_DIR / f"{stem}.npz"
        if not npz_path.exists():
            rows.append({"video_stem": stem, "note": "npz不在"})
            continue
        _, defrag, _ = _process_video(npz_path, sim, 0)
        cand = [e for e in defrag if e.fire_side == side]
        if not cand:
            rows.append({"video_stem": stem, "note": "該当side イベントなし"})
            continue
        target = min(cand, key=lambda e: abs(e.t_chain_start - t_cs_approx))
        old_dur_rel = target.t_fire - target.t_chain_start
        new_dur_rel = chain_completion_from_formula(0.0, target.chain_count)
        cbin = min(target.chain_count, CHAIN_BIN_CAP)
        pop_row = bins.loc[cbin] if cbin in bins.index else None
        rows.append({
            "video_stem": stem, "fire_side": side,
            "chain_count": target.chain_count, "frag_count": target.frag_count,
            "old_tfire_based_dur_sec": round(old_dur_rel, 2),
            "new_formula_based_dur_sec": round(new_dur_rel, 2),
            "pop_visual_median_sec": None if pop_row is None else round(float(pop_row["visual_median"]), 2),
            "pop_visual_min_sec": None if pop_row is None else round(float(pop_row["visual_min"]), 2),
            "pop_visual_max_sec": None if pop_row is None else round(float(pop_row["visual_max"]), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[定数] CHAIN_ANIM_PER_STEP_SEC = {CHAIN_ANIM_PER_STEP_SEC} 秒/連鎖 (user指定値)")
    print("\n=== (1) 母集団 chain_count ビン別 比較 (t_chain_start=0 とした相対秒) ===")
    print(
        "⚠️ formula_appear_sec の実測値が無いため t_chain_start を近似として使用 "
        "(過小評価の可能性を含む近似値であることに注意)"
    )
    pop = _population_bin_comparison()
    print(pop.to_string())

    print("\n=== (2) 外れ値6件 (user提示) の新旧比較 ===")
    out = _outlier_case_check()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
