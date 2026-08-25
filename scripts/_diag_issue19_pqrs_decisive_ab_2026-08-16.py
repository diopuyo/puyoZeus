"""指摘19: 安全弁 (--resolved-kill-override) 自体が不要かの決定的A/B (read-only)。

coordinator依頼 (2026-08-16): 案1(strict)が正しく直った今、安全弁は不要
(誤爆リスクだけが残る) 可能性を検証する。4構成 P/Q/R/S を同一2窓・
フレーム単位 (sample_interval=0、fps=30換算で約33ms刻み、要求の0.1秒以下
を満たす) で比較する。production_config.py は一切変更しない (数値を出す
だけ)。

  P: strict ON / kill_override OFF / counter_aware -- (安全弁なし)
  Q: strict ON / kill_override ON  / counter_aware OFF (現行の採用構成)
  R: strict ON / kill_override ON  / counter_aware ON  (今回実装の状態ゲート)
  S: strict OFF/ kill_override OFF / counter_aware -- (指摘14修正前、対照)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
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
    layout="panel", render=False,
)

CONFIGS = {
    "P_strict_only": dict(
        enable_resolved_live_defender_strict=True,
        enable_resolved_kill_override=False,
        enable_resolved_kill_override_counter_aware=False),
    "Q_strict+kill(現行)": dict(
        enable_resolved_live_defender_strict=True,
        enable_resolved_kill_override=True,
        enable_resolved_kill_override_counter_aware=False),
    "R_strict+kill+gate(今回)": dict(
        enable_resolved_live_defender_strict=True,
        enable_resolved_kill_override=True,
        enable_resolved_kill_override_counter_aware=True),
    "S_none(修正前対照)": dict(
        enable_resolved_live_defender_strict=False,
        enable_resolved_kill_override=False,
        enable_resolved_kill_override_counter_aware=False),
}

results: dict[str, list[tuple[float, float]]] = {}
for name, kw in CONFIGS.items():
    hist: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, Path(f"data/verify/_unused_issue19_pqrs_{name}.mp4"),
        debug_history_out=hist, **BASE, **kw,
    )
    results[name] = hist
    print(f"[run] {name} 完了 ({len(hist)}サンプル)")


def _dump_window(lo: float, hi: float, label: str) -> None:
    print(f"\n=== {label} (t={lo}-{hi}秒、フレーム単位 ~33ms刻み) ===")
    names = list(CONFIGS.keys())
    header = "t_abs   " + "".join(f"{n[:18]:>20}" for n in names)
    print(header)
    # P系列の時刻を基準に、各構成の最近傍サンプルを突き合わせる
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


_dump_window(194.5, 201.0, "窓1(指摘14、正しい致死判定)")
_dump_window(200.0, 207.0, "窓2(指摘19、誤った致死判定)")
print("\nDIAG_ISSUE19_PQRS_DONE")
