"""エフェクト被りセルのHSV特性を実測し「発光検知」の実現可能性を確認する。

設計案: 発光中のセルは分類器が COLOR_UNKNOWN を返す
→ 既存の _merge_diff_only D分岐 (UNKNOWN→baseline維持) が自動で働く
→ エフェクトが晴れたら正しい色で確定する (保留するだけ、消さない)。

成立条件: 「被りセル」と「正常セル」がHSV統計で分離できること。
被りセル (v3ラベルから):
  011 c15 2P f18294: r1c0,r1c1,r1c3,r2c1,r2c3 (相殺の黄光, 真値=赤)
  015 c17 2P f89724: r2c1 (紫球, 真値=黄)
  032 c12 1P f125902: r2c2 (×印の赤光, 真値=緑)
対照: 同じフレームの正常認識セル。
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.image_reader import DEFAULT_P1_REGION as P1, DEFAULT_P2_REGION as P2

def stats(f, reg, r, c):
    x1,y1,x2,y2 = reg.cell_sample_rect(r,c)
    p = f[y1:y2, x1:x2]
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    blown = float(((V>=240)&(S<=80)).mean())     # 白飛び率
    bright = float((V>=230).mean())               # 高輝度率
    return (int(np.median(H)), int(np.median(S)), int(np.median(V)), blown, bright)

cases = [
    ("video_c15", P2, 18294, [(1,0),(1,1),(1,3),(2,1),(2,3)], [(8,0),(10,0),(12,2),(11,4)]),
    ("video_c17", P2, 89724, [(2,1)], [(4,1),(7,2),(10,3)]),
    ("video_c12", P1, 125902, [(2,2),(1,3),(2,3)], [(6,1),(9,2),(11,4)]),
]
print(f"{'種別':<6}{'セル':>8}{'H':>5}{'S':>5}{'V':>5}{'白飛び':>8}{'高輝度':>8}")
for vid, reg, fr, bad, good in cases:
    cap = cv2.VideoCapture(f"data/frames/{vid}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fr)); ok, f = cap.read(); cap.release()
    if f.shape[:2] != (1080,1920): f = cv2.resize(f,(1920,1080),interpolation=cv2.INTER_AREA)
    print(f"--- {vid} f{fr} ---")
    for (r,c) in bad:
        h,s,v,bl,br = stats(f,reg,r,c)
        print(f"{'被り':<6}{f'r{r}c{c}':>8}{h:>5}{s:>5}{v:>5}{bl:>8.2f}{br:>8.2f}")
    for (r,c) in good:
        h,s,v,bl,br = stats(f,reg,r,c)
        print(f"{'正常':<6}{f'r{r}c{c}':>8}{h:>5}{s:>5}{v:>5}{bl:>8.2f}{br:>8.2f}")
