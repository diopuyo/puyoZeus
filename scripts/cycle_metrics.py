"""サイクル評価メトリクス計算スクリプト.

各 viz log + viz mp4 から以下を集計:
  - constraint-mismatch 件数 (CNN 過剰検出)
  - state 分布 (STABLE 比率)
  - override / vote_updated 発火回数
  - 色変化推定: 同 cell の色が頻繁に変化 = 不安定 = 誤認 signal

cycle 間の比較に使う。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def analyze_log(log_path: Path) -> dict:
    """viz log を集計."""
    if not log_path.exists():
        return {"missing": True}
    text = log_path.read_text(errors="ignore")
    mismatch_count = text.count("[constraint-mismatch]")
    constraint_count = text.count("[constraint] ")
    online_hsv_count = text.count("online_hsv injected")
    progress_lines = re.findall(
        r"\[progress\] (\d+)/(\d+) .* 1P=(\w+) 2P=(\w+)",
        text,
    )
    p1_states: dict = {}
    p2_states: dict = {}
    last_frames = 0
    for f, n, p1, p2 in progress_lines:
        p1_states[p1] = p1_states.get(p1, 0) + 1
        p2_states[p2] = p2_states.get(p2, 0) + 1
        last_frames = int(n)
    sample_n = len(progress_lines)
    p1_stable_pct = (
        p1_states.get("stable", 0) / sample_n * 100 if sample_n else 0
    )
    p2_stable_pct = (
        p2_states.get("stable", 0) / sample_n * 100 if sample_n else 0
    )
    done = "[done]" in text
    return {
        "missing": False,
        "log": str(log_path),
        "n_frames": last_frames,
        "sample_n": sample_n,
        "mismatch_count": mismatch_count,
        "constraint_replaced_count": constraint_count,
        "online_hsv_inject_count": online_hsv_count,
        "p1_stable_pct": round(p1_stable_pct, 1),
        "p2_stable_pct": round(p2_stable_pct, 1),
        "done": done,
    }


def storage_total_mb(directory: Path) -> float:
    """指定 directory 配下の mp4 合計サイズ (MB)."""
    total = 0
    for p in directory.rglob("*.mp4"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total / 1024 / 1024


def main() -> int:
    import json
    root = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
    if not root.exists():
        # Windows path のフォールバック
        root = Path("C:/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
    logs_dir = root / "logs"
    test_unknown_dir = root / "data" / "test_unknown"
    out: dict = {
        "test_unknown_size_mb": round(
            storage_total_mb(test_unknown_dir), 1,
        ),
        "logs": {},
    }
    log_patterns = sys.argv[1:] if len(sys.argv) > 1 else [
        "viz_v97m11_v3_*.log", "viz_v70_v3_*.log",
    ]
    for pat in log_patterns:
        for log in logs_dir.glob(pat):
            out["logs"][log.name] = analyze_log(log)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
