"""修正①適用後に新規発生した14エピソードの実画面を抽出する (2026-08-22)。

読み取り専用 (元動画 data/frames/video_zenchi_c0BQoMJwwQU.mp4 を読むだけ、
納品物 data/verify/zenchi_delivery_2026-08-21 には一切触れない)。
幅800pxにリサイズしたJPEGで出力する (スマホ確認用)。
"""
import subprocess
from pathlib import Path

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT_DIR = Path("logs/_new_episodes_evidence_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 新規14件 (代表時刻=中央値、_compare_kill_override_fix_episodes の出力より)
TARGETS = [
    (205.3, "seg01_a"),
    (311.45, "seg01_b"),
    (817.97, "seg01_c"),
    (1165.07, "seg02_a"),
    (1961.13, "seg03_a"),
    (3478.65, "seg04_a"),
    (3970.8, "seg05_a"),
    (4362.5, "seg05_b"),
    (4365.15, "seg05_c"),
    (4362.5, "seg06_a_dup_of_seg05_b"),
    (4364.3, "seg06_b"),
    (4585.67, "seg06_c"),
    (4800.83, "seg06_d"),
    (5671.95, "seg07_a"),
]

for t, name in TARGETS:
    out_jpg = OUT_DIR / f"{name}_t{t:.2f}.jpg"
    subprocess.run(
        [FF, "-y", "-ss", str(t), "-i", str(SRC), "-frames:v", "1",
         "-vf", "scale=800:-1", "-q:v", "3", str(out_jpg)],
        check=True, capture_output=True,
    )
    print(f"{name}: t={t:.3f} -> {out_jpg}")

print("done")
