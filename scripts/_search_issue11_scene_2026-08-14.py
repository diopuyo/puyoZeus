"""指摘11 (旧デモ2v4 t=37-38、着弾前空白の誤判定) の該当シーンを、
再生成後のdemo2 (video_74) タイムラインdumpから再特定する。

1試合目相当 (source 230-286s) で pending (飛来予告) が大きい区間を抜き出し、
「小連鎖の受け側が有利表示になっていないか」を確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import load_timeline_dump

DUMP = Path("data/verify/demo_fixed_2026-08-13/selfverify_demo2v2_fullscan_2026-08-14.npz")


def main() -> int:
    vid, rows = load_timeline_dump(DUMP)
    for r in rows:
        if not (230.0 <= r.t_sec <= 407.0):
            continue
        if (r.pending_p1 >= 15 or r.pending_p2 >= 15) and r.state1 != "CHAIN" and r.state2 != "CHAIN":
            print(f"t={r.t_sec:.2f} p1={r.p1*100:.1f}% score1={r.score1} score2={r.score2} "
                  f"pend1={r.pending_p1} pend2={r.pending_p2} "
                  f"state1={r.state1} state2={r.state2} top1={r.drivers_top1_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
