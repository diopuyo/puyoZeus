"""指摘14 (2026-08-15) 診断: final3_m1 (162-218秒) で表示された「19%」の
根因追跡。

診断対象: 本線同士の打ち合い後、片側 (会計上 pend2=216 まで確認済み、
scripts/_diag_issue13_verify_2026-08-15.py が生成した dump より) が未着弾
おじゃまで実質詰み状態にもかかわらず、勝率19%が表示された経緯。

本体コード (scripts/visualize_advantage_overlay.py, src/*) は一切変更しない。
モジュールを import し ov.generate() を呼ぶだけ (monkeypatch もしない)。
debug_history_out (実際の表示値 disp_adv、毎フレーム) と
dump_timeline_path (settled 更新時のみ、room/pending/drivers 等の内訳) の
2つを重ね合わせて、19%がどの経路 (ライブ per-frame / resolved hold /
live_defender_reeval) から出たかを特定する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

START_SEC = 162.0
END_SEC = 219.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue14.mp4")
DUMP = Path("logs/diag_issue14_scene_2026-08-15.npz")


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
        dump_timeline_path=DUMP,
    )
    print(f"\n[history] n={len(history)} rows (実際の表示値、毎フレーム)")
    print("t_sec(絶対)  disp_adv   p1(1P)%   p2(2P)%")
    last_bucket = None
    for t_abs, disp_adv in history:
        bucket = round(t_abs / 0.2)
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        p1 = ov.adv_to_winprob(disp_adv)
        print(f"  t={t_abs:7.2f} disp_adv={disp_adv:+7.2f} "
              f"p1={p1 * 100:5.1f}% p2={100 - p1 * 100:5.1f}%")

    print("\n[dump] settled 更新記録 (room/pending/drivers 内訳)")
    _video_id, rows = load_timeline_dump(DUMP)
    print("t_sec     p1_raw%  p1(disp相当)%  pend1  pend2  room1  room2  "
          "drv_top1")
    for r in rows:
        print(f"  t={r.t_sec:7.2f} p1_raw={r.p1_raw*100:5.1f} p1={r.p1*100:5.1f} "
              f"pend1={r.pending_p1:4d} pend2={r.pending_p2:4d} "
              f"room1={r.room1:3d} room2={r.room2:3d} "
              f"drv={r.drivers_top1_name}:{r.drivers_top1_val:+.2f}")
    print("\nDONE_ISSUE14_SCENE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
