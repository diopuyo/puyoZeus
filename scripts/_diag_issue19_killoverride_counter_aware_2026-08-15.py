"""指摘19対処 (enable_kill_override_counter_aware) の A/B/C 検証 (read-only)。

3構成を同一窓で比較する:
  A) 安全弁OFF                    (enable_resolved_kill_override=False)
  B) 安全弁ON (現行、指摘19が出る)  (enable_resolved_kill_override=True,
                                    counter_aware=False)
  C) 安全弁ON + 新フラグON (修正後) (enable_resolved_kill_override=True,
                                    counter_aware=True)

合格条件:
  1. t=201.4-203.4 の誤った「1P 0.7%」が消える (Cが56%前後、またはOFF寄り)
  2. t=195.8-200.8 の正しい致死判定 99.3% は維持される (C も 99%台を維持)

さらに指摘14窓 (194.53-201秒) と指摘13窓 (234.87-245.5秒) の非退行も確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
BASE = dict(
    max_sec=0.0, sample_interval=0.0, start_sec=162.0, end_sec=250.0,
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
    layout="panel", render=False,
)

CONFIGS = {
    "A_kill_OFF": dict(enable_resolved_kill_override=False,
                        enable_resolved_kill_override_counter_aware=False),
    "B_kill_ON": dict(enable_resolved_kill_override=True,
                       enable_resolved_kill_override_counter_aware=False),
    "C_kill_ON_counter_aware": dict(enable_resolved_kill_override=True,
                                     enable_resolved_kill_override_counter_aware=True),
}

results: dict[str, list[tuple[float, float]]] = {}
for name, kw in CONFIGS.items():
    hist: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, Path(f"data/verify/_unused_issue19_{name}.mp4"),
        debug_history_out=hist, **BASE, **kw,
    )
    results[name] = hist
    print(f"[run] {name} 完了 ({len(hist)}サンプル)")


def _p1_at(name: str, t: float, tol: float = 0.05) -> "float | None":
    cand = [a for tt, a in results[name] if abs(tt - t) < tol]
    if not cand:
        return None
    return ov.adv_to_winprob(cand[0]) * 100.0


def _print_window(lo: float, hi: float, label: str) -> None:
    print(f"\n=== {label} (t={lo}-{hi}秒、0.25秒刻み) ===")
    print("t_abs     A_OFF     B_ON      C_counter_aware")
    seen = set()
    for t, _ in results["A_kill_OFF"]:
        if not (lo <= t <= hi):
            continue
        b = round(t * 4)
        if b in seen:
            continue
        seen.add(b)
        pa = _p1_at("A_kill_OFF", t)
        pb = _p1_at("B_kill_ON", t)
        pc = _p1_at("C_kill_ON_counter_aware", t)
        fmt = lambda v: f"{v:6.1f}%" if v is not None else "  ----"
        print(f"{t:7.2f}  {fmt(pa)}   {fmt(pb)}   {fmt(pc)}")


_print_window(195.8, 200.8, "指摘14窓(致死は維持されるべき)")
_print_window(200.0, 207.0, "指摘19窓(誤爆区間)")
_print_window(194.53, 201.0, "指摘14 回帰確認窓")
_print_window(234.87, 245.5, "指摘13 回帰確認窓")

print("\nDIAG_ISSUE19_COUNTER_AWARE_DONE")
