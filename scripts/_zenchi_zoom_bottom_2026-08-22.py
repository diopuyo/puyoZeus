import cv2
from pathlib import Path
img = cv2.imread("data/verify/_layout_check_2026-08-22/set1_minadv_t890.png")
# グラフ領域 y:792-1080, 右端付近(カーソル周辺) x:1250-1408 を拡大
crop = img[792:1080, 1200:1408, :]
crop_big = cv2.resize(crop, (crop.shape[1]*3, crop.shape[0]*3), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("data/verify/_layout_check_2026-08-22/crops/set1_minadv_t890_cursor_zoom.png", crop_big)
# 最下段の数行のピクセル値をプリント (赤バーがy=1079まで届いているか)
import numpy as np
col_range = img[1070:1080, 1290:1310, :]
print("y=1070..1079, x=1290..1310 の色 (B,G,R):")
for yy in range(col_range.shape[0]):
    print(1070+yy, col_range[yy].mean(axis=0))
