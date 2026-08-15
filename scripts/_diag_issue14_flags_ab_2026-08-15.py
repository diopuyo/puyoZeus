"""指摘14 (2026-08-15) 案1/案2 フラグの回帰窓 ON/OFF 数値比較。

対象窓: 絶対 t=194.53-201秒 (final3_m1 で 2P に誤って生存率18.9%が5.2秒表示
された区間、scripts/_diag_issue14_scene_2026-08-15.py と同じ動画/条件)。

4通りを同一パイプラインで生成し history (毎フレーム disp_adv) を比較する:
  A) baseline: enable_resolved_live_defender_strict=False, enable_resolved_kill_override=False
     (指摘13までの既存挙動、既定値のまま)
  B) +strict:  enable_resolved_live_defender_strict=True のみ
  C) +kill:    enable_resolved_kill_override=True のみ
  D) +both:    両方 True

本体コード (scripts/visualize_advantage_overlay.py) は一切変更しない
(monkeypatch もしない、generate() の既存 kwargs のみで比較する)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

START_SEC = 162.0
END_SEC = 219.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14_ab.mp4")
WINDOW = (194.53, 201.0)  # 指摘14の誤爆窓 (絶対秒)

COMBOS = {
    "A_baseline": dict(enable_resolved_live_defender_strict=False,
                       enable_resolved_kill_override=False),
    "B_strict": dict(enable_resolved_live_defender_strict=True,
                     enable_resolved_kill_override=False),
    "C_kill": dict(enable_resolved_live_defender_strict=False,
                   enable_resolved_kill_override=True),
    "D_both": dict(enable_resolved_live_defender_strict=True,
                  enable_resolved_kill_override=True),
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
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        **extra,
    )
    return history


def main() -> int:
    results: dict[str, list[tuple[float, float]]] = {}
    for name, extra in COMBOS.items():
        print(f"[run] {name} extra={extra}")
        results[name] = _run(extra)

    print(f"\n[window] t={WINDOW[0]:.2f}-{WINDOW[1]:.2f}秒 の disp_adv/p1(1P)% 比較")
    header = "  t_abs   | " + " | ".join(f"{n:>22s}" for n in COMBOS)
    print(header)
    base_hist = results["A_baseline"]
    seen_buckets = set()
    for t_abs, _ in base_hist:
        if not (WINDOW[0] <= t_abs <= WINDOW[1]):
            continue
        bucket = round(t_abs / 0.5)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        row = [f"t={t_abs:6.2f}"]
        for name in COMBOS:
            hist = results[name]
            # 同一フレーム数列のはずなので同じ index で引く (t が一致する前提)。
            match = next((adv for (t2, adv) in hist if abs(t2 - t_abs) < 1e-6), None)
            if match is None:
                row.append(" " * 22)
                continue
            p1 = ov.adv_to_winprob(match)
            row.append(f"adv={match:+6.1f} p1={p1*100:5.1f}%")
        print("  " + " | ".join(row))

    print("\nDONE_ISSUE14_FLAGS_AB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
