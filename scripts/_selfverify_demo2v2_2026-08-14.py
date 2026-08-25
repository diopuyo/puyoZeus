"""検収セルフベリファイ (デモ2 = demo2v2, video_74 3試合、未見動画):
指摘11 (着弾前空白の誤判定) の非退行確認を --no-render 単発実行で行う (read-only)。

_gen_demo2_2026-08-13.sh (HEAD 78426de、指摘12決着後) と完全同一のフラグ
構成 (--resolved-decisive-amplify は付けない、元スクリプト通り) で
generate(render=False) を実行し、以下を出力する:
  - 指摘11窓 (source 264-272s、旧デモ2 v4 の t=37-38 相当): 表示値 disp_adv
    の 0.2秒刻みサンプル。1Pが2連鎖で対応中に約5段のおじゃまが飛来する
    局面で、着弾前に1P有利と誤判定していないかを確認する。
  - 全域チェック: dump_timeline から極端値割合・主因分布を集計。

本体コード (scripts/visualize_advantage_overlay.py) は変更しない。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

START_SEC = 230.0
END_SEC = 407.0
VIDEO = Path("data/frames/video_74.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_selfverify_demo2v2.mp4")
DUMP_PATH = Path("data/verify/demo_fixed_2026-08-13/selfverify_demo2v2_fullscan_2026-08-14.npz")

ISSUE11_LO, ISSUE11_HI = 262.0, 272.0


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
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        dump_timeline_path=DUMP_PATH,
    )

    print(f"n_history_samples={len(history)}")

    print("\n===== 指摘11窓 (source 262-272s、旧デモ2v4 t=37-38相当): 表示値 disp_adv (0.2秒刻み) =====")
    last_bucket = None
    for t_sec, disp_adv in history:
        if not (ISSUE11_LO <= t_sec <= ISSUE11_HI):
            continue
        bucket = round(t_sec * 5) / 5.0
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = ov.adv_to_winprob(disp_adv)
        print(f"t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} p1(1P)={p1*100:5.1f}% "
              f"p2(2P)={100 - p1*100:5.1f}%")

    print("\n===== 全域チェック (dump_timeline, source 230-407s) =====")
    video_id, rows = load_timeline_dump(DUMP_PATH)
    n_total = len(rows)
    n_extreme_p1 = sum(1 for r in rows if r.p1 >= 0.97)
    n_extreme_p2 = sum(1 for r in rows if r.p1 <= 0.03)
    print(f"video_id={video_id} n_settled_updates={n_total}")
    print(f"極端値(p1>=97%): {n_extreme_p1}/{n_total} ({n_extreme_p1/n_total*100:.1f}%)")
    print(f"極端値(p1<=3%): {n_extreme_p2}/{n_total} ({n_extreme_p2/n_total*100:.1f}%)")
    driver_counter = Counter(r.drivers_top1_name for r in rows)
    print("主因(top1) 列名分布 (上位10):")
    for name, cnt in driver_counter.most_common(10):
        print(f"  {name:30s} {cnt:5d} ({cnt/n_total*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
