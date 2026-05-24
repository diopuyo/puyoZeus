"""eval_frame_0900s.png の 1P 盤面を CNN で読み取って詳細をダンプ。
sanity 違反 'main_chain_maturity=0.00' の原因を特定する。
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

from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
from src.calibration import CalibratedConfig
from src.chain import ChainSimulator
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier

CHAR = {0: "・", 1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡", 5: "🟣", 9: "💥"}


def dump_board(grid: np.ndarray, tag: str) -> None:
    print(f"\n--- {tag} (grid {grid.shape}) ---")
    # 隠し段を含めて表示
    for r in range(grid.shape[0]):
        row_str = "".join(CHAR.get(int(grid[r, c]), "?") for c in range(grid.shape[1]))
        mark = " <HIDDEN" if r < HIDDEN_ROWS else ""
        print(f"  row{r:2d}: {row_str}{mark}")


def main() -> None:
    cnn = CnnPatchClassifier.load(Path("models/cnn_best.pt"))
    config = CalibratedConfig.load("models/calibration_video01.json")
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(classifier=gated, p1_region=config.p1_region, p2_region=config.p2_region)
    calc = IndicatorCalculator()

    frame_path = Path("data/verify/eval_cycle/eval_frame_0900s.png")
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"読み込み失敗: {frame_path}")
        return
    print(f"frame: {frame.shape}")

    b1, b2 = reader.read_both_boards(frame)
    dump_board(b1._grid, "1P")
    dump_board(b2._grid, "2P")

    sim = ChainSimulator()
    for tag, b in [("1P", b1), ("2P", b2)]:
        res = sim.simulate(b)
        print(f"\n{tag} simulate: chain={res.chain_count} erased={res.total_erased} puyos={b.count_puyos()}")

    # 指標
    print("\n--- 1P 指標 ---")
    i1 = calc.compute_all(b1)
    for name, r in i1.results.items():
        print(f"  {name}: {r.score:+.4f}")
    print("\n--- 2P 指標 ---")
    i2 = calc.compute_all(b2)
    for name, r in i2.results.items():
        print(f"  {name}: {r.score:+.4f}")


if __name__ == "__main__":
    main()
