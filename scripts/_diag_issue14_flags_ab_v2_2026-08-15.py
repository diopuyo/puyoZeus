"""指摘14 案1やり直し後の A/B 再検証 (2026-08-15、coordinator指示 手順3/4)。

state ベースに修正した strict (`enable_resolved_live_defender_strict`) が:
  (3) 指摘14の誤爆窓 (絶対194.53-201秒) で 96.1% 付近を維持できるか
      (kill_override 無しの strict 単体で退行が消えるか)
  (4) 指摘13の正当ケース窓 (絶対234.87-245.5秒) で、正当な0.5秒ごとの
      再評価が strict によって止まっていないか (指摘13の効果喪失チェック)

を同一動画・同一設定 (kill_override は両方 OFF、strict のみ ON/OFF で比較)
の1本の generate() 呼び出しで両窓を同時に確認する (2回で済ませ、
_diag_issue14_flags_ab_2026-08-15.py で確認済みの kill_override 単体効果は
再検証しない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

START_SEC = 162.0
END_SEC = 248.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14_ab_v2.mp4")
WINDOW_14 = (194.53, 201.0)   # 指摘14 誤爆窓
WINDOW_13 = (234.87, 245.5)  # 指摘13 正当ケース窓

COMBOS = {
    "A_baseline_strict_off": dict(enable_resolved_live_defender_strict=False),
    "B_strict_on_state_based": dict(enable_resolved_live_defender_strict=True),
}


def _run(extra: dict) -> list[tuple[float, float]]:
    history: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC,
        show_recognition=True,
        enable_early_fire_reaction=True, enable_per_side_settled=True,
        disable_score_lead_bias=True, disable_pressure=True,
        enable_counter_remaining_time=True, enable_counter_defender_only=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=True,
        enable_resolved_kill_override=False,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        **extra,
    )
    return history


def _print_window(results: dict[str, list[tuple[float, float]]], window: tuple[float, float],
                  label: str) -> None:
    print(f"\n[window {label}] t={window[0]:.2f}-{window[1]:.2f}秒")
    header = "  t_abs   | " + " | ".join(f"{n:>26s}" for n in COMBOS)
    print(header)
    base_hist = results["A_baseline_strict_off"]
    seen_buckets = set()
    for t_abs, _ in base_hist:
        if not (window[0] <= t_abs <= window[1]):
            continue
        bucket = round(t_abs / 0.5)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        row = [f"t={t_abs:6.2f}"]
        for name in COMBOS:
            hist = results[name]
            match = next((adv for (t2, adv) in hist if abs(t2 - t_abs) < 1e-6), None)
            if match is None:
                row.append(" " * 26)
                continue
            p1 = ov.adv_to_winprob(match)
            row.append(f"adv={match:+7.1f} p1={p1*100:5.1f}%")
        print("  " + " | ".join(row))


def main() -> int:
    results: dict[str, list[tuple[float, float]]] = {}
    for name, extra in COMBOS.items():
        print(f"[run] {name} extra={extra}")
        results[name] = _run(extra)

    _print_window(results, WINDOW_14, "指摘14誤爆窓")
    _print_window(results, WINDOW_13, "指摘13正当ケース窓")

    print("\nDONE_ISSUE14_FLAGS_AB_V2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
