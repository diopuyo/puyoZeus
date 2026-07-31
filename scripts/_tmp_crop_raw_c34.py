"""raw フレームから 2P ROI をクロップし拡大保存する (読み取り専用診断用)。"""
from PIL import Image

# 1920x1080 換算 ROI 定数 (scripts/visualize_recognition.py と同一値) を
# 実フレーム解像度 1280x720 にスケール (2/3) して使う。
SCALE = 1280 / 1920
P2_ROI_X = int(1258 * SCALE)
P2_ROI_Y = int(160 * SCALE)
ROI_W = int(384 * SCALE)
ROI_H = int(720 * SCALE)

for stem in ["raw_468.8", "raw_470.1", "raw_473.1"]:
    im = Image.open(f"/tmp/c34raw/{stem}.png")
    crop = im.crop((P2_ROI_X, P2_ROI_Y, P2_ROI_X + ROI_W, P2_ROI_Y + ROI_H))
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    crop.save(f"/tmp/c34raw/{stem}_2Pcrop.png")
    print(stem, im.size, "-> crop", (P2_ROI_X, P2_ROI_Y, P2_ROI_X + ROI_W, P2_ROI_Y + ROI_H))
