"""指摘19 根治 (--resolved-victim-gen-live) の合格判定 A/B (read-only)。

4構成を同一2窓・0.1秒以下刻み (sample_interval=0.0、fps=30換算で約33ms) で
比較する。production_config.py は一切変更しない (数値を出すだけ)。

  BASE   : 現行採用構成 (strict+kill+counter_aware、victim_gen_live OFF)
  FIX    : BASE + --resolved-victim-gen-live ON
  NOGATE : FIX から counter_aware を OFF に戻す (根治だけで安全弁ガード無しでも
           直るかを見る、参考値)
  OFF_ALL: 指摘14修正前の対照 (strict OFF、対照)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
BASE = dict(
    max_sec=0.0, sample_interval=0.0,
    show_recognition=True,
    enable_early_fire_reaction=True, enable_per_side_settled=True,
    disable_score_lead_bias=True, disable_pressure=True,
    enable_counter_remaining_time=True, enable_counter_defender_only=True,
    enable_ojama_fall_placement_override=True,
    enable_resolved_exchange_eval=True,
    enable_resolved_decisive_amplify=True,
    enable_pseudo_chain_score_fill=True,
    enable_resolved_live_defender=True,
    enable_resolved_live_defender_strict=True,
    enable_resolved_kill_override=True,
    layout="panel", render=False,
)

CONFIGS = {
    "BASE(現行採用)": dict(
        enable_resolved_kill_override_counter_aware=True,
        enable_resolved_victim_gen_live=False),
    "FIX(根治ON)": dict(
        enable_resolved_kill_override_counter_aware=True,
        enable_resolved_victim_gen_live=True),
    "NOGATE(根治のみ)": dict(
        enable_resolved_kill_override_counter_aware=False,
        enable_resolved_victim_gen_live=True),
}

# 窓1/2 (162-210秒) と窓3 (230-246秒、指摘13既存合格窓) は離れているため
# 別々に処理して履歴を結合する (動画全体を処理する無駄を避ける)。
RANGES = [(162.0, 210.0), (230.0, 246.0)]

results: dict[str, list[tuple[float, float]]] = {}
for name, kw in CONFIGS.items():
    hist: list[tuple[float, float]] = []
    for start_sec, end_sec in RANGES:
        ov.generate(
            VIDEO, Path(f"data/verify/_unused_issue19_victimlive_{name}.mp4"),
            start_sec=start_sec, end_sec=end_sec,
            debug_history_out=hist, **BASE, **kw,
        )
    results[name] = hist
    print(f"[run] {name} 完了 ({len(hist)}サンプル)")


def _dump_window(lo: float, hi: float, label: str) -> None:
    print(f"\n=== {label} (t={lo}-{hi}秒、フレーム単位 ~33ms刻み) ===")
    names = list(CONFIGS.keys())
    header = "t_abs   " + "".join(f"{n[:18]:>20}" for n in names)
    print(header)
    base_ts = [t for t, _ in results[names[0]] if lo <= t <= hi]
    for t in base_ts:
        row = f"{t:6.2f}  "
        for n in names:
            cand = [a for tt, a in results[n] if abs(tt - t) < 0.02]
            if cand:
                p = ov.adv_to_winprob(cand[0]) * 100.0
                row += f"{p:19.1f}%"
            else:
                row += f"{'----':>20}"
        print(row)


_dump_window(194.5, 201.0, "窓1(指摘14、正しい致死判定、合格=99.3%維持)")
_dump_window(201.2, 203.4, "窓2(指摘19、誤った致死判定、合格=56%前後)")
_dump_window(234.87, 245.5, "窓3(指摘13、既存合格窓、非退行確認)")
print("\nDIAG_ISSUE19_VICTIM_GEN_LIVE_AB_DONE")
