import cv2
import numpy as np
img = cv2.imread("data/verify/_layout_check_2026-08-22/set1_maxadv_t479.png")
graph = img[792:1080, 0:1408, :]
b, g, r = graph[:,:,0].astype(int), graph[:,:,1].astype(int), graph[:,:,2].astype(int)
blue_mask = (b > 80) & (b > r + 20) & (b > g)
rows = np.where(blue_mask.any(axis=1))[0]
print("青ピクセル行範囲(相対):", rows.min(), rows.max(), "絶対y:", 792+rows.min(), 792+rows.max())
col_top = []
for c in range(graph.shape[1]):
    rr = np.where(blue_mask[:, c])[0]
    if len(rr):
        col_top.append(rr.min())
col_top = np.array(col_top)
print("列ごとの青最上行の最小値(相対):", col_top.min(), " 絶対y:", 792+col_top.min())
print("ボックス上端(想定絶対y): 792")
