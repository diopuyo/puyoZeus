"""検収①: D1a/D1b 疑義エピソードの実画面フレーム抽出 (読み取り専用)。"""
import subprocess
from pathlib import Path

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21")
OUT_DIR = Path("data/verify/_scan_evidence_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SET_BOUNDARY = 3626.0

TARGETS = [
    (200.467, "d1b_t200_2Pdeath_ignored"),
    (887.000, "d1b_t887_2Pdeath_ignored_severe"),
    (4896.000, "d1a_t4896_2Pdead_ignored"),
    (3227.967, "d1b_t3228_1Pdeath_ignored"),
    (16.500, "d1a_t16_2Pdead_ignored_earliest"),
]

for t_global, name in TARGETS:
    if t_global < SET_BOUNDARY:
        video = "zenchi_set1_audio.mp4"
        t_local = t_global
    else:
        video = "zenchi_set2_audio.mp4"
        t_local = t_global - SET_BOUNDARY
    src = SRC / video
    out_png = OUT_DIR / f"{name}_t{t_global:.1f}.png"
    subprocess.run([FF, "-y", "-ss", str(t_local), "-i", str(src), "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)
    print(f"{name}: {video} local_t={t_local:.3f} -> {out_png}")

print("done")
