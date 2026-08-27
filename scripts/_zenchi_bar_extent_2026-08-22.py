import cv2
import numpy as np
img = cv2.imread("data/verify/_layout_check_2026-08-22/set1_minadv_t890.png")
graph = img[792:1080, 0:1408, :]  # y:0..287 相対
# 赤系ピクセル (B<G<R程度、しきい値ラフ) を検出し、各列で最も下にある行を探す
b, g, r = graph[:,:,0].astype(int), graph[:,:,1].astype(int), graph[:,:,2].astype(int)
red_mask = (r > 80) & (r > g + 20) & (r > b + 20)
rows_with_red = np.where(red_mask.any(axis=1))[0]
print("赤ピクセルが存在する行範囲(グラフ相対 y):", rows_with_red.min(), "-", rows_with_red.max(), "  (グラフ高さ=", graph.shape[0], ")")
print("絶対y座標:", 792+rows_with_red.min(), "-", 792+rows_with_red.max())
print("グラフボックス絶対下端(想定): 792+288=1080 (最終行1079)")
# 各列ごとの赤最下行
col_bottom = []
for c in range(graph.shape[1]):
    rr = np.where(red_mask[:, c])[0]
    if len(rr):
        col_bottom.append(rr.max())
col_bottom = np.array(col_bottom)
print("列ごとの赤最下行の最大値(グラフ相対):", col_bottom.max() if len(col_bottom) else None)
print("そのときの絶対y:", 792+col_bottom.max() if len(col_bottom) else None)

# 青(1P advantage)も同様に最上行を確認 (別フレームだが同じ画像内にmax_advの履歴はない。ここは-100フレームなのでスキップ)
