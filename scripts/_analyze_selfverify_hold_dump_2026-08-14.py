"""検収セルフベリファイ: ホールド区間dump (npz) からt=34-38秒(出力相対)の
settled再計算イベント列を表示する (read-only)。

出力t = source_t - 230 (start_sec)。dump の t_sec は source 絶対時刻。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_advantage_overlay import load_timeline_dump

START_SEC = 230.0
WINDOW_LO = 32.0  # 出力相対 (少し広めに見る)
WINDOW_HI = 40.0


def main() -> int:
    path = Path("data/verify/demo_fixed_2026-08-13/demo2_v4_selfverify_hold_dump.npz")
    video_id, rows = load_timeline_dump(path)
    print(f"video_id={video_id} n_rows={len(rows)}")
    for r in rows:
        t_out = r.t_sec - START_SEC
        if WINDOW_LO <= t_out <= WINDOW_HI:
            print(
                f"t_out={t_out:6.2f}s (src={r.t_sec:7.2f}) p1={r.p1*100:5.1f}% "
                f"state1={r.state1:12s} state2={r.state2:12s} "
                f"score1={r.score1} score2={r.score2} adv_raw={r.adv_raw:.3f} "
                f"adv_ema={r.adv_ema:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
