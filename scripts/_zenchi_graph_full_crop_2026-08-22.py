import cv2
from pathlib import Path
SRC_DIR = Path("data/verify/_layout_check_2026-08-22")
OUT_DIR = SRC_DIR / "crops"
for name in ["set1_t1800", "set2_t2800"]:
    img = cv2.imread(str(SRC_DIR / f"{name}.png"))
    graph_full = img[780:1080, 0:1408, :]
    cv2.imwrite(str(OUT_DIR / f"{name}_graph_full.png"), graph_full)
print("done")
