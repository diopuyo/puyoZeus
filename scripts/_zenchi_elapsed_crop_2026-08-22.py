import cv2
from pathlib import Path
d = Path("data/verify/_elapsed_glitch_check_2026-08-22")
for name in ["before_split_t893", "after_split_t894"]:
    img = cv2.imread(str(d / f"{name}.png"))
    crop = img[1000:1080, 1408:1920]
    cv2.imwrite(str(d / f"{name}_elapsed_crop.png"), crop)
print("done")
