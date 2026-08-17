"""指摘13 forecast整合修正 (2026-08-15) の再検証用スクリプト。

初版検証 (_diag_issue13_verify_2026-08-15.py) で、受け側の盤面だけをライブ
(着弾前) に差し替えた際、forecast (お邪魔予告) が leftover のみのままで
dropped_to_pX 分を静かに見落とし、脅威を過小評価する事象を発見した
(ResolvedExchangeTracker._live_defender_snap で修正済み)。

本スクリプトは ON (enable_resolved_live_defender=True) のみを、指摘12/指摘10
両窓をカバーする最小限の範囲 (162-245s、指摘12/#9対応後の148s全域より短い)
で再実行し、修正後の値が過剰に振れていないか (依然として1P有利へ完全反転
していないか) を確認する。OFF は初版検証+既存ユニットテストで bit-identical
を既に確認済みのため本スクリプトでは実行しない (処理コスト削減)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402

START_SEC = 162.0
END_SEC = 245.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue13_fix.mp4")

ISSUE12_LO, ISSUE12_HI = 234.87, 243.0
ISSUE10_LO, ISSUE10_HI = 194.53, 200.0


def _bucketed(history: list[tuple[float, float]], lo: float, hi: float,
              bucket_sec: float = 0.5) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    last_bucket = None
    for t_sec, disp_adv in history:
        if not (lo <= t_sec <= hi):
            continue
        bucket = round(t_sec / bucket_sec)
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        out.append((t_sec, disp_adv))
    return out


def _print_timeline(label: str, rows: list[tuple[float, float]]) -> None:
    print(f"\n---- {label} ----")
    for t_sec, disp_adv in rows:
        p1 = 0.5 + disp_adv / 200.0
        print(f"  t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} "
              f"p1(1P)={p1 * 100:5.1f}% p2(2P)={100 - p1 * 100:5.1f}%")


def main() -> int:
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
    )

    print("\n===== 指摘12窓 (source 234.87-243s): disp_adv 0.5秒刻み (修正後ON) =====")
    _print_timeline("ON (forecast整合修正後)", _bucketed(history, ISSUE12_LO, ISSUE12_HI))

    print("\n===== 指摘10窓 (source 194.53-200s): disp_adv 0.5秒刻み (修正後ON) =====")
    _print_timeline("ON (forecast整合修正後)", _bucketed(history, ISSUE10_LO, ISSUE10_HI))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
