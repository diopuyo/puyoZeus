"""検収③: グラフ/右パネルのレイアウト検証用フレーム抽出+ピクセル分析。
納品物(zenchi_set1_audio.mp4 / zenchi_set2_audio.mp4)を読み取り専用で扱う。
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
OUT_DIR = Path("data/verify/_layout_check_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("set1_audio.mp4", 30, "set1_t0030"),
    ("set1_audio.mp4", 1800, "set1_t1800"),
    ("set1_audio.mp4", 3600, "set1_t3600"),
    ("set2_audio.mp4", 30, "set2_t0030"),
    ("set2_audio.mp4", 2800, "set2_t2800"),
    ("set2_audio.mp4", 3390, "set2_t3390"),
]

SRC = Path("data/verify/zenchi_delivery_2026-08-21")


def extract_frame(video: str, t: float, out_png: Path) -> None:
    src = SRC / f"zenchi_{video}"
    cmd = [FF, "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1", str(out_png)]
    subprocess.run(cmd, check=True, capture_output=True)


def analyze(png: Path) -> dict:
    img = cv2.imread(str(png))
    h, w = img.shape[:2]
    assert (h, w) == (1080, 1920), f"想定外の解像度: {w}x{h}"
    result = {}
    # 4隅 (5x5ピクセル平均)
    corners = {
        "top_left": img[0:5, 0:5],
        "top_right": img[0:5, w-5:w],
        "bottom_left": img[h-5:h, 0:5],
        "bottom_right": img[h-5:h, w-5:w],
    }
    for k, v in corners.items():
        result[f"corner_{k}_bgr_mean"] = v.reshape(-1, 3).mean(axis=0).tolist()
    # 下端ライン全体 (y=1079) の黒率
    bottom_row = img[1079:1080, :, :]
    black_mask = (bottom_row < 10).all(axis=2)
    result["bottom_row_black_ratio"] = float(black_mask.mean())
    # グラフ想定領域 (左カラム下部 x:0-1407 y:792-1079) の最下段行の黒率
    graph_bottom = img[1078:1080, 0:1408, :]
    graph_black = (graph_bottom < 10).all(axis=2)
    result["graph_area_bottom2px_black_ratio"] = float(graph_black.mean())
    # 右パネル (x:1408-1919) の下端5行の黒率 (文字が生きていれば黒率100%にはならない想定域だが、
    # 単純に「真っ黒帯」の有無だけをまず見る)
    right_bottom = img[1075:1080, 1408:1920, :]
    right_black = (right_bottom < 10).all(axis=2)
    result["right_panel_bottom5px_black_ratio"] = float(right_black.mean())
    return result


def main() -> int:
    for video, t, name in TARGETS:
        png = OUT_DIR / f"{name}.png"
        extract_frame(video, t, png)
        stats = analyze(png)
        print(f"--- {name} (t={t}) ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
