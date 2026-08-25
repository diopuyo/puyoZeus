"""物差しv2ラベル監査の疑いセルに赤枠を付けたROI画像を書き出す (2026-08-17)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._crop_label_audit_evidence_2026-08-17
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
MANIFEST_PATH = YARDSTICK_DIR / "manifest.json"
OUT_DIR = _ROOT / "data" / "verify" / "yardstick_v2_label_audit_2026-08-17" / "suspect_frames"

# (sheet_id, [(r,c), ...])
TARGETS: list[tuple[str, list[tuple[int, int]]]] = [
    ("026_c13_2P_f17462", [(9, 3)]),  # 既知r9c3 (クロスチェック用)
    ("043_c22_1P_f108676", [(12, 2)]),  # 既知c22 (クロスチェック用)
    ("019_c23_1P_f150153", [(11, 0), (12, 0), (12, 1), (4, 0), (4, 1)]),  # 新規候補+未triaged残り2セル(枠色で区別)
    ("058_c21_2P_f143682", [(5, 1), (6, 1)]),  # 新規候補
    ("033_c13_2P_f91334", [(4, 2)]),  # 既知の測定器誤集計(a)、対照として撮る
    ("001_c109_2P_f66674", [(3, 3)]),  # 未triaged残り
    ("028_c16_2P_f42831", [(5, 4)]),  # 未triaged残り
    ("034_c11_1P_f10330", [(6, 0), (6, 1)]),  # 未triaged残り
    ("004_c21_1P_f144486", [(4, 0)]),  # 行修正後に新たに弱信号で浮上
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {e["sheet_id"]: e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
    for sheet_id, cells in TARGETS:
        e = manifest[sheet_id]
        roi_path = _ROOT / e["board_roi_png"]
        img = cv2.imread(str(roi_path))
        if img is None:
            print(f"[warn] 読み込み失敗: {roi_path}")
            continue
        h, w = img.shape[:2]
        # board_roi_png は隠し段(行0)を含まない可視12行のみ (image_reader.DEFAULT_P1/
        # P2_REGION height=720 / 12行 = 60px/行)。ラベル行r(1〜12)は画像行(r-1)に
        # オフセットする (診断スクリプトのバグとして2026-08-17に発見・修正)。
        cell_h, cell_w = h / 12.0, w / 6.0
        # 全セルグリッド線を薄く描く
        vis = img.copy()
        for r in range(13):
            y = int(r * cell_h)
            cv2.line(vis, (0, y), (w, y), (80, 80, 80), 1)
        for c in range(7):
            x = int(c * cell_w)
            cv2.line(vis, (x, 0), (x, h), (80, 80, 80), 1)
        for (r, c) in cells:
            y0, y1 = int((r - 1) * cell_h), int(r * cell_h)
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
        out_path = OUT_DIR / f"{sheet_id}_marked.png"
        cv2.imwrite(str(out_path), vis)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
