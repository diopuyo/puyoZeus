"""検収④: 試合境界の目視突合用フレーム抽出 (読み取り専用)。"""
import subprocess
from pathlib import Path

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21")
OUT_DIR = Path("data/verify/_boundary_check_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SET_BOUNDARY = 3626.0

TARGETS = [
    (3625.0, "boundary_set1end_before"),
    (3626.5, "boundary_set2start_after"),
    (1797.5, "midset_boundary_before"),
    (1799.5, "midset_boundary_after"),
]

for t_global, name in TARGETS:
    if t_global < SET_BOUNDARY:
        video, t_local = "zenchi_set1_audio.mp4", t_global
    else:
        video, t_local = "zenchi_set2_audio.mp4", t_global - SET_BOUNDARY
    out_png = OUT_DIR / f"{name}_t{t_global:.1f}.png"
    subprocess.run([FF, "-y", "-ss", str(t_local), "-i", str(SRC / video), "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)
    print(name, "->", out_png)
