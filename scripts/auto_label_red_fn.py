"""
赤の偽陰性候補（HSV 上明確に赤なのに CNN が非赤と判定したパッチ）を
自動的に赤ラベル付きの jsonl に書き出す。

安全策:
    - red_ratio >= RATIO_THRESHOLD のみ採用 (0.50 以上で相当強い赤)
    - 人手ラベル互換形式 (kind=correction, patch_file 絶対パス) で出力

使い方:
    ./venv/bin/python scripts/auto_label_red_fn.py
    ./venv/bin/python scripts/auto_label_red_fn.py --threshold 0.60

出力:
    data/verify/human_labels/auto_red_fn_<ts>.jsonl
    data/verify/human_labels/auto_red_fn_<ts>/patches/*.png
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_PURPLE,
    COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.calibration import CalibratedConfig
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from scripts.extract_red_fn import _red_ratio, TARGET_CLASSES  # 再利用

DEFAULT_CNN: Path = Path("models/cnn_global_best.pt")
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")
HUMAN_LABELS_DIR: Path = Path("data/verify/human_labels")

CODE_TO_STR: dict[int, str] = {
    COLOR_EMPTY: "empty", COLOR_RED: "red", COLOR_BLUE: "blue",
    3: "green", COLOR_YELLOW: "yellow", COLOR_PURPLE: "purple", 9: "ojama",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="赤 hue 比率がこの値以上なら赤として採用")
    args = parser.parse_args()

    cnn = CnnPatchClassifier.load(DEFAULT_CNN)
    config = CalibratedConfig.load(DEFAULT_CALIB)
    gated = GatedCnnClassifier(color_classifier=cnn)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = HUMAN_LABELS_DIR / f"auto_red_fn_{ts}"
    patches_dir = out_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = HUMAN_LABELS_DIR / f"auto_red_fn_{ts}.jsonl"

    sample_dir = Path("data/frames/sample")
    frames = sorted(p for p in sample_dir.glob("frame_*.png") if "debug" not in p.name)

    n_kept = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        meta = {
            "kind": "meta",
            "import_ts": ts,
            "tool": "auto_label_red_fn.py",
            "threshold": args.threshold,
            "policy": "HSV 赤 hue 比率が閾値以上で非赤 CNN 予測のセルを赤に訂正",
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for fp in frames:
            frame = cv2.imread(str(fp))
            if frame is None or frame.shape[:2] != (1080, 1920):
                continue
            for side_name, region in (("1P", config.p1_region), ("2P", config.p2_region)):
                for row in range(HIDDEN_ROWS, BOARD_ROWS):
                    for col in range(BOARD_COLS):
                        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2c <= x1c or y2c <= y1c:
                            continue
                        patch = frame[y1c:y2c, x1c:x2c].copy()
                        pred = gated.classify(patch)
                        if pred not in TARGET_CLASSES:
                            continue
                        ratio = _red_ratio(patch)
                        if ratio < args.threshold:
                            continue
                        # 採用: 赤として記録
                        patch_rel = f"patches/{fp.stem}_{side_name}_{row:02d}_{col:02d}.png"
                        patch_abs = (out_root / patch_rel).resolve()
                        cv2.imwrite(str(patch_abs), patch)
                        entry = {
                            "kind": "correction",
                            "frame": str(fp),
                            "side": side_name,
                            "row": row,
                            "col": col,
                            "cnn_predicted": CODE_TO_STR.get(pred, str(pred)),
                            "true_label": "red",
                            "patch_file": str(patch_abs),
                            "auto_labeled": True,
                            "red_ratio": round(ratio, 3),
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        n_kept += 1

    print(f"自動赤ラベル完了: {jsonl_path}")
    print(f"  採用: {n_kept} パッチ (閾値 {args.threshold})")
    print(f"  パッチ: {patches_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
