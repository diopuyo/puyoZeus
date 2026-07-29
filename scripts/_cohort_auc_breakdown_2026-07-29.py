"""LOGO内訳ログから中盤AUCを動画群(c20/m20/m30)別に集計する。

自動生成summaryの「combined40から悪化=新動画の質の問題」という判定が
妥当かを検証する。新規26本(m30)の中盤AUCが既存46本より系統的に低いなら
質の問題、同等なら単なる分散。
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

LOG = Path(
    "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/"
    "data/verify/win_eval_combined66_2026-07-29/combined66_video_breakdown.log"
)

# 動画番号 → コホート
def cohort(num: int) -> str:
    if num <= 33:
        return "c20 (c10-c29)"
    if num <= 55:
        return "m20 (c34-c55)"
    return "m30 (c56-c81) 新規"


def main() -> None:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    # スコープごとにブロック分割
    blocks = re.split(r"スコープ:\s*(\S+)", text)
    # blocks = [前置き, name1, body1, name2, body2, ...]
    for i in range(1, len(blocks) - 1, 2):
        scope = blocks[i]
        body = blocks[i + 1]
        rows: dict[str, list[float]] = {}
        for m in re.finditer(r"video_c(\d+)\s+\d+.*?([01]\.\d{4})\s*$", body, re.M):
            num, auc = int(m.group(1)), float(m.group(2))
            rows.setdefault(cohort(num), []).append(auc)
        if not rows:
            continue
        print(f"\n=== スコープ: {scope} ===")
        for key in sorted(rows):
            v = rows[key]
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            se = sd / (len(v) ** 0.5) if v else 0.0
            print(
                f"  {key:22s} n={len(v):2d}  平均AUC={statistics.mean(v):.4f}"
                f"  中央={statistics.median(v):.4f}  std={sd:.4f}  SE={se:.4f}"
            )
        allv = [x for v in rows.values() for x in v]
        print(f"  {'全66本':22s} n={len(allv):2d}  平均AUC={statistics.mean(allv):.4f}")


if __name__ == "__main__":
    main()
