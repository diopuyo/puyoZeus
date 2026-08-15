"""検収で発見した反転 (t=201.2〜202.8 で 1P 0.7% 表示) の原因特定 (read-only)。

疑い: --resolved-kill-override が「受け側がまだ撃ち返せる」局面で致死断定して
いる。ルール上おじゃまは受け側のツモ着地時に降るため、受け側は先に撃ち返せる。
安全弁のみ ON/OFF して同一窓を比較する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
LO, HI = 200.0, 207.0
BASE = dict(
    max_sec=0.0, sample_interval=0.0, start_sec=162.0, end_sec=210.0,
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

results: dict[str, list[tuple[float, float]]] = {}
for name, kill in (("kill_ON", True), ("kill_OFF", False)):
    hist: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, Path(f"data/verify/demo_final4_2026-08-15/_unused_{name}.mp4"),
        enable_resolved_kill_override=kill, debug_history_out=hist, **BASE,
    )
    results[name] = [(t, a) for t, a in hist if LO <= t <= HI]
    print(f"[run] {name} 完了 ({len(results[name])}サンプル)")

print(f"\n=== t={LO}-{HI}秒 1P勝率 比較 (0.25秒刻み) ===")
print("t_abs   kill_ON   kill_OFF   差")
seen = set()
for t, a_on in results["kill_ON"]:
    b = round(t * 4)
    if b in seen:
        continue
    seen.add(b)
    cand = [x for x in results["kill_OFF"] if abs(x[0] - t) < 0.05]
    p_on = ov.adv_to_winprob(a_on) * 100.0
    if cand:
        p_off = ov.adv_to_winprob(cand[0][1]) * 100.0
        print(f"{t:6.2f}  {p_on:6.1f}%   {p_off:6.1f}%   {p_on - p_off:+6.1f}")
    else:
        print(f"{t:6.2f}  {p_on:6.1f}%      ----")
print("DIAG_ISSUE19_DONE")
