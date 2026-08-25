import subprocess, cv2
from pathlib import Path
FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21/zenchi_set1_audio.mp4")
OUT = Path("data/verify/_elapsed_glitch_check_2026-08-22")
for t in [896, 900, 910, 930, 960]:
    out_png = OUT / f"t{t}.png"
    subprocess.run([FF, "-y", "-ss", str(t), "-i", str(SRC), "-frames:v", "1", str(out_png)], check=True, capture_output=True)
    img = cv2.imread(str(out_png))
    crop = img[1000:1080, 1408:1920]
    cv2.imwrite(str(OUT / f"t{t}_elapsed_crop.png"), crop)
print("done")
