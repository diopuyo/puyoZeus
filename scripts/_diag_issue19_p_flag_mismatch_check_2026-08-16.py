"""P(strict単体)が72.1%(現行final4相当flag)と96.1%(過去指摘14 v2 A/B)で
食い違う件の原因切り分け (read-only)。

過去 A/B (_diag_issue14_flags_ab_v2_2026-08-15.py) は stable_majority_window
/ ojama_fall_entry_hardening / ojama_fall_scoped_exit の3フラグを含んで
いたが、これらは同日8/15付けで「不採用確定」となり final4 デモ生成
スクリプト (_gen_demo_final4_2026-08-15.sh) では外されている。この3フラグ
を足し戻すと96.1%を再現するか (=測定器としての整合性確認) を検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
BASE = dict(
    max_sec=0.0, sample_interval=0.0, start_sec=162.0, end_sec=202.0,
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
    enable_resolved_kill_override=False,
    enable_resolved_kill_override_counter_aware=False,
    layout="panel", render=False,
)

CONFIGS = {
    "P_final4相当(不採用3フラグ無し)": dict(),
    "P_旧v2相当(不採用3フラグ有り)": dict(
        stable_majority_window=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
    ),
}

results: dict[str, list[tuple[float, float]]] = {}
for name, kw in CONFIGS.items():
    hist: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, Path(f"data/verify/_unused_issue19_pcheck_{name}.mp4"),
        debug_history_out=hist, **BASE, **kw,
    )
    results[name] = hist
    print(f"[run] {name} 完了 ({len(hist)}サンプル)")

print("\n=== 窓1(指摘14、194.5-201.0秒) 比較 ===")
names = list(CONFIGS.keys())
base_ts = [t for t, _ in results[names[0]] if 195.6 <= t <= 200.9]
for t in base_ts:
    row = f"{t:6.2f}  "
    for n in names:
        cand = [a for tt, a in results[n] if abs(tt - t) < 0.02]
        if cand:
            p = ov.adv_to_winprob(cand[0]) * 100.0
            row += f"{p:8.1f}%"
    print(row)

print("\nDIAG_ISSUE19_P_FLAG_MISMATCH_DONE")
