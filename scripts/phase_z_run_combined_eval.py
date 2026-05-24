"""v16+HSV+anomaly の組み合わせ cross_video 評価。

OnlineHsvCalibrator + CellAnomalyDetector の両方を有効化して評価し、
v16/v17b/v16+HSV と比較する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_run_combined_eval
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_z_cross_video",
        "--duration", "30",
        "--cnn-model", "models/cnn_phase_u_v16.pt",
        "--use-online-hsv",
        "--use-cell-anomaly",
        "--out-suffix", "v16_hsv_anomaly",
    ]
    env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        cmd, cwd=str(_ROOT), env=env, check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
