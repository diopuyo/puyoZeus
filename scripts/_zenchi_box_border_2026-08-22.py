import cv2
import numpy as np
img = cv2.imread("data/verify/_layout_check_2026-08-22/set1_maxadv_t479.png")
graph = img[792:1080, 0:1408, :]
# 白っぽい横線(枠線)を検出: 明るいグレー/白 (B,G,R全て高め かつ均一)
gray = graph.mean(axis=2)
# 各行の平均輝度が高い行(横罫線候補)を出す
row_mean = gray.mean(axis=1)
bright_rows = np.where(row_mean > 40)[0]
print("明るい行(相対y, 上位20件):", bright_rows[:20])
print("明るい行(相対y, 下位20件):", bright_rows[-20:])
for rr in bright_rows:
    print(rr, 792+rr, row_mean[rr])
