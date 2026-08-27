"""検収用: 指定フレームの1P/2P盤面領域を切り出し拡大 (read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

IN_PATH = Path(sys.argv[1])
OUT_PATH = Path(sys.argv[2])
# x1,y1,x2,y2
x1, y1, x2, y2 = (int(v) for v in sys.argv[3:7])
scale = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0

img = cv2.imread(str(IN_PATH))
crop = img[y1:y2, x1:x2]
h, w = crop.shape[:2]
resized = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_PATH), resized)
print(f"[saved] {OUT_PATH}")
