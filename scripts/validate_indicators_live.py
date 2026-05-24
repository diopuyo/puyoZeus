"""data/frames/sample の 1920x1080 生フレームで CNN + 修正済み indicators/scorer を通し、
1P/2P 指標・総合スコアを一覧出力する。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.calibration import CalibratedConfig
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.scorer import Scorer


def main() -> None:
    # holdout best を保護した cnn_global_best.pt を優先する (学習中 cnn_best.pt は劣化版になりうる)
    cnn_path = Path("models/cnn_global_best.pt")
    if not cnn_path.exists():
        cnn_path = Path("models/cnn_best.pt")
    cnn = CnnPatchClassifier.load(cnn_path)
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(classifier=gated, p1_region=config.p1_region, p2_region=config.p2_region)
    calc = IndicatorCalculator()
    scorer = Scorer()

    sample_dir = Path("data/frames/sample")
    # 生フレームのみ (frame_NNNNs.png)
    frames = sorted(p for p in sample_dir.glob("frame_*.png") if "debug" not in p.name)
    print(f"対象: {len(frames)}フレーム\n")

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None or img.shape[:2] != (1080, 1920):
            print(f"{fp.name}: スキップ (shape={None if img is None else img.shape})")
            continue
        try:
            b1, b2 = reader.read_both_boards(img)
        except Exception as e:
            print(f"{fp.name}: 読み取り例外 {e}")
            continue

        n1, n2 = b1.count_puyos(), b2.count_puyos()
        i1 = calc.compute_all(b1)
        i2 = calc.compute_all(b2)
        res = scorer.score(i1, i2)

        print(f"=== {fp.name}  1P={n1:>2}  2P={n2:>2}  total={res.total_score:+6.1f} ({res.advantage_side()}) ===")
        print("  指標        |   1P      2P    diff")
        print("  ------------|------ ------ ------")
        for name in i1.results:
            s1 = i1.results[name].score
            s2 = i2.results[name].score
            print(f"  {name:<12}| {s1:+5.2f}  {s2:+5.2f}  {s1 - s2:+5.2f}")
        print()


if __name__ == "__main__":
    main()
