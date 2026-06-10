"""
既存モデルの簡易評価 — 盤面読み取り結果をテキスト出力。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
from src.calibration import CalibratedConfig
from src.image_reader import ImageReader
from src.old.indicators import IndicatorCalculator
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.old.scorer import Scorer

LBL = {0: "_", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", 9: "O"}


def eval_model(model_path: Path, frames_dir: Path, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"モデル: {label} ({model_path.name})")
    print('=' * 60)

    if not model_path.exists():
        print(f"  モデルなし")
        return

    cnn = CnnPatchClassifier.load(model_path)
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(classifier=gated, p1_region=config.p1_region, p2_region=config.p2_region)
    calc = IndicatorCalculator()
    scorer = Scorer()

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"  フレームなし: {frames_dir}")
        return

    for fp in frames[:6]:
        if "debug" in fp.name:
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        try:
            b1, b2 = reader.read_both_boards(frame)
            i1 = calc.compute_all(b1)
            i2 = calc.compute_all(b2)
            result = scorer.score(i1, i2)
            # 盤面表示 (可視段のみ)
            visible_start = HIDDEN_ROWS
            b1_vis = b1._grid[visible_start:]
            b2_vis = b2._grid[visible_start:]
            # 非空セル数
            nonzero_1p = int(np.sum(b1_vis != 0))
            nonzero_2p = int(np.sum(b2_vis != 0))
            print(f"\n{fp.name}:")
            print(f"  score={result.total_score:+6.1f} ({result.advantage_side()})")
            print(f"  1P セル数: {nonzero_1p}/72  2P セル数: {nonzero_2p}/72")
            print(f"  1P 指標: {', '.join(f'{n}={i1.score_of(n):+.2f}' for n in list(i1.results.keys())[:4])}")
            print(f"  2P 指標: {', '.join(f'{n}={i2.score_of(n):+.2f}' for n in list(i2.results.keys())[:4])}")
        except Exception as e:
            print(f"  {fp.name}: エラー {e}")


def main() -> None:
    print("##### 既存モデル簡易評価 #####")

    frames_dir = Path("data/verify/eval_cycle")
    if not frames_dir.exists():
        frames_dir = Path("data/frames/sample")

    # 6ch 形式のみ (cnn_eval_cycle がそれ)
    models_to_check = [
        ("models/cnn_eval_cycle.pt", "cnn_eval_cycle (最新)"),
    ]

    for path_str, label in models_to_check:
        try:
            eval_model(Path(path_str), frames_dir, label)
        except Exception as e:
            print(f"  [スキップ] {path_str}: {e}")


if __name__ == "__main__":
    main()
