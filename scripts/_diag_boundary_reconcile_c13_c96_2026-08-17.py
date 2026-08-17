"""W22根治検証: c13 / c96 でも境界マルチシグナルの事後救済
(_reconcile_boundary_anomalies) の一致率を測る (本体コード非変更)。

ncc_scan_2026-08-17 の score_zero both_zero=1 の孤立サンプル時刻を
「実際の試合開始候補」の近似アンカーとし、その前後の短い窓だけを
collect_lean(enable_boundary_multisignal=True) で処理して、
`<out>_boundary_anomalies.json` (事後救済後) の件数を確認する。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_boards_lean import collect_lean  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ncc_scan_c13.csv / ncc_scan_c96.csv の sz_both_zero=1 孤立サンプル時刻
# (5秒間隔サンプリングでの近似値、±20秒の窓を取る)。
WINDOWS = [
    ("c13", "data/frames/video_c13.mp4", 535.0, 40.0),
    ("c13", "data/frames/video_c13.mp4", 1405.0, 40.0),
    ("c96", "data/frames/video_c96.mp4", 1070.0, 40.0),
    ("c96", "data/frames/video_c96.mp4", 1480.0, 40.0),
    ("c96", "data/frames/video_c96.mp4", 3485.0, 40.0),
]


def main() -> int:
    results = []
    for name, video, start_sec, max_sec in WINDOWS:
        video_path = Path(video)
        if not video_path.exists():
            print(f"[{name}@{start_sec}] MISSING: {video_path}", flush=True)
            continue
        tag = f"{name}_{int(start_sec)}"
        out_npz = OUT_DIR / f"reconcile_{tag}.npz"
        t0 = time.time()
        n = collect_lean(
            video_path, out_npz,
            start_sec=start_sec, max_sec=max_sec,
            enable_boundary_multisignal=True,
        )
        dt = time.time() - t0
        anomaly_path = out_npz.with_name(out_npz.stem + "_boundary_anomalies.json")
        n_anomalies = 0
        if anomaly_path.exists():
            n_anomalies = len(json.loads(anomaly_path.read_text(encoding="utf-8")))
        print(
            f"[{tag}] snapshots={n} anomalies_after_reconcile={n_anomalies} "
            f"elapsed={dt:.1f}s",
            flush=True,
        )
        results.append((tag, n, n_anomalies))
    print("SUMMARY:", flush=True)
    for tag, n, n_anom in results:
        print(f"  {tag}: snapshots={n} anomalies_after_reconcile={n_anom}", flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
