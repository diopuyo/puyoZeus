"""②③関連の副産物調査: 「経過」カウンタがセグメント境界(893.7s等、字幕/ロード無し)を
跨ぐ瞬間に不連続に見えるかを実フレームで確認する (読み取り専用)。"""
import subprocess
from pathlib import Path

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21")
OUT_DIR = Path("data/verify/_elapsed_glitch_check_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# t=893.7 は set1 内部のセグメント境界 (ロード画面なし、連続ゲームプレイのはず)
TARGETS = [
    (893.0, "before_split_t893"),
    (894.5, "after_split_t894"),
]

for t_local, name in TARGETS:
    out_png = OUT_DIR / f"{name}.png"
    subprocess.run([FF, "-y", "-ss", str(t_local), "-i", str(SRC / "zenchi_set1_audio.mp4"),
                     "-frames:v", "1", str(out_png)], check=True, capture_output=True)
    print(name, "->", out_png)
