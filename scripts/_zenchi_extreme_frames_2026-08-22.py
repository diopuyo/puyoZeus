"""検収③: グラフの上端/下端到達を確認するための極値フレーム抽出。"""
import subprocess
from pathlib import Path
import cv2

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21")
OUT_DIR = Path("data/verify/_layout_check_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("set1_audio.mp4", 479.23, "set1_maxadv_t479"),
    ("set1_audio.mp4", 890.87, "set1_minadv_t890"),
]

for video, t, name in TARGETS:
    src = SRC / f"zenchi_{video}"
    out_png = OUT_DIR / f"{name}.png"
    subprocess.run([FF, "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)
    img = cv2.imread(str(out_png))
    graph_full = img[780:1080, 0:1408, :]
    cv2.imwrite(str(OUT_DIR / "crops" / f"{name}_graph_full.png"), graph_full)

print("done")
