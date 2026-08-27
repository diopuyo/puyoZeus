"""検収③: レイアウト目視確認用クロップ画像を生成する (読み取り専用)。"""
import cv2
from pathlib import Path

SRC_DIR = Path("data/verify/_layout_check_2026-08-22")
OUT_DIR = SRC_DIR / "crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["set1_t0030", "set1_t1800", "set1_t3600", "set2_t0030", "set2_t2800", "set2_t3390"]

for name in NAMES:
    img = cv2.imread(str(SRC_DIR / f"{name}.png"))
    h, w = img.shape[:2]
    # 下端帯 (y:950-1080, 全幅) — グラフ下端+字幕帯残り確認用
    bottom_strip = img[950:1080, :, :]
    cv2.imwrite(str(OUT_DIR / f"{name}_bottom_strip.png"), bottom_strip)
    # 右パネル全体 (x:1408-1920, y:0-1080) — 経過時刻行クリップ確認用
    right_panel = img[:, 1408:1920, :]
    cv2.imwrite(str(OUT_DIR / f"{name}_right_panel.png"), right_panel)
    # 4隅 (各200x200)
    tl = img[0:200, 0:200]
    tr = img[0:200, w-200:w]
    bl = img[h-200:h, 0:200]
    br = img[h-200:h, w-200:w]
    cv2.imwrite(str(OUT_DIR / f"{name}_corner_tl.png"), tl)
    cv2.imwrite(str(OUT_DIR / f"{name}_corner_tr.png"), tr)
    cv2.imwrite(str(OUT_DIR / f"{name}_corner_bl.png"), bl)
    cv2.imwrite(str(OUT_DIR / f"{name}_corner_br.png"), br)

print("done")
