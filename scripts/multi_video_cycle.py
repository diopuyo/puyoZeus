"""複数動画で同じ cycle を回して汎用性を検証.

各 cycle:
  - 5 video viz 生成
  - 各 viz の metrics 集計
  - cycle 横並び比較レポート
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 5 動画 (= 多様な codec / 品質 / 学習 status)
TEST_VIDEOS: list[tuple[str, Path]] = [
    ("v97", Path("data/evaluation_videos/v97_match11_96s.mp4")),       # unseen h264 high
    ("v70", Path("data/evaluation_videos/v70_match2_113s.mp4")),       # trained mpeg4 high
    ("v89m3", Path("data/evaluation_videos/v89_match3_95s.mp4")),      # trained h264 low
    ("v50", Path("data/test_unknown/v50_match1_75s_720p.mp4")),        # trained h264 med
    ("v91", Path("data/test_unknown/v91_match1_75s_720p.mp4")),        # trained h264 med
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cycle", type=int, required=True)
    p.add_argument("--cnn-model", type=str,
                   default="models/cnn_phase_b_large_v3.pt")
    p.add_argument("--parallel", type=int, default=2,
                   help="並列 viz 数 (GPU 共有のため 2-3 推奨)")
    p.add_argument("--cnn-override-prob", type=float, default=None,
                   help="HybridClassifier の CNN 採用閾値. None で hybrid_classifier の default")
    p.add_argument("--hsv-state", type=str, default=None,
                   help="OnlineHsv の事前注入 JSON path (e.g. data/per_video_hsv_ranges/_merged_default.json)")
    args = p.parse_args()

    out_dir = Path("data/test_unknown")
    log_dir = Path("logs")
    venv_py = str(_ROOT / "venv" / "bin" / "python")

    procs: list[subprocess.Popen] = []
    for tag, vid in TEST_VIDEOS:
        if not vid.exists():
            print(f"[skip] {vid} not found")
            continue
        out = out_dir / f"{tag}_viz_multicycle_{args.cycle}.mp4"
        log = log_dir / f"viz_{tag}_multicycle_{args.cycle}.log"
        cmd = [
            venv_py, "-m", "scripts.visualize_recognition",
            "--video", str(vid),
            "--output", str(out),
            "--cnn-model", args.cnn_model,
        ]
        if args.cnn_override_prob is not None:
            cmd.extend(["--cnn-override-prob", str(args.cnn_override_prob)])
        if args.hsv_state is not None:
            cmd.extend(["--hsv-state", args.hsv_state])
        env = {**os.environ, "PYTHONPATH": str(_ROOT)}
        log_f = open(log, "w")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(_ROOT))
        procs.append((tag, p, log, out))
        print(f"[launched] {tag} -> {out}")
        # 並列制限
        while sum(1 for _, pr, _, _ in procs if pr.poll() is None) >= args.parallel:
            time.sleep(5)

    # 全完了待ち
    print(f"[waiting] {len(procs)} viz processes...")
    for tag, pr, log, out in procs:
        rc = pr.wait()
        print(f"[done] {tag} rc={rc} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
