"""スマホ版レビューツールのビルダー (2026-08-05、使い捨て)。

シート画像を縮小JPEG化してbase64データURIを生成し、
data/verify/mobile_review_2026-08-05/images.json に書き出す。
HTML本体は別途 (main Claude が組み立てる)。
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

SHEET_DIR = Path("data/verify/error_onset_sheet_2026-08-04")
OUT_DIR = Path("data/verify/mobile_review_2026-08-05")
TARGET_WIDTH = 720
JPEG_QUALITY = 62

# レビューで見せるシーンとシート
TARGETS = {
    "c18_scene": "sheet_c18_01.png",   # 10セル同時 (最大の改善例 10→2)
    "c19_scene": "sheet_c19_01.png",   # 着弾汚染 22セル (22→0)
    "c15_scene": "sheet_c15_01.png",   # 7→0
    "c29_scene": "sheet_c29_01.png",   # 残存例 (弱光帯)
}


CROP_TOP_ROWS = 4  # シート上位N行のみ (1行=1セルの3コマ組、代表例で十分)
SHEET_ROWS = 10


def encode(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    row_h = img.height // SHEET_ROWS
    if row_h > 0 and img.height > row_h * CROP_TOP_ROWS:
        img = img.crop((0, 0, img.width, row_h * CROP_TOP_ROWS))
    if img.width > TARGET_WIDTH:
        h = int(img.height * TARGET_WIDTH / img.width)
        img = img.resize((TARGET_WIDTH, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    total = 0
    for key, fname in TARGETS.items():
        p = SHEET_DIR / fname
        if not p.exists():
            print(f"欠落: {p}")
            continue
        uri = encode(p)
        out[key] = uri
        total += len(uri)
        print(f"{key}: {len(uri) // 1024} KB")
    (OUT_DIR / "images.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"合計 {total // 1024} KB -> {OUT_DIR / 'images.json'}")


if __name__ == "__main__":
    main()
