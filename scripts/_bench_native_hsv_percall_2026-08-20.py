"""Rust HSV 分類を「パッチ1枚ずつ呼ぶ」形での実効速度を測る (2026-08-20)。

`_bench_native_hsv_2026-08-20.py` は盤面まるごと1回 (境界越え1回、cvtColor は
事前計算) で 9.4倍を出したが、**本番の read_board は既にセルパッチを切り出して
いる**ため、そのまま配線すると「パッチごとに Rust を呼ぶ」形になる。
この形では
  - Python/Rust 境界を 78 回越える
  - cvtColor もパッチごとに 1 回ずつ走る (現行と同じ回数)
ので、9.4倍はそのまま出ない。配線方式を決めるために実効値を測る。

比較:
  A. Python 現行 (`ColorClassifier.classify`、cvtColor 込み)
  B. Rust パッチ毎呼び出し (cvtColor + 境界越え 込み)
  C. Rust 盤面まるごと (参考: cvtColor 1回 + 境界越え1回)

C が大きく勝つなら、read_board 側で ROI+rects を組み直す配線 (工数増) の
価値がある。B で十分なら最小変更で済む。
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

_spec = importlib.util.spec_from_file_location(
    "_parity_helpers",
    PROJECT_ROOT / "scripts" / "_verify_native_hsv_parity_2026-08-20.py",
)
_ph = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ph)

from src.image_reader import ColorClassifier  # noqa: E402

_CELLS = 78  # 13x6
_CELL_PX = 16


def main() -> int:
    """3 方式の 1 盤面あたり所要を比較する。"""
    import puyo_core

    clf = ColorClassifier()
    ranges = _ph._ranges_flat(clf)
    params = _ph._params(clf)
    rng = np.random.default_rng(20260820)

    patches = [
        np.ascontiguousarray(
            rng.integers(0, 256, (_CELL_PX, _CELL_PX, 3), dtype=np.uint8)
        )
        for _ in range(_CELLS)
    ]
    # 盤面まるごと版の入力 (13x6 を 1 枚の ROI に並べたもの)
    roi = np.ascontiguousarray(
        rng.integers(0, 256, (_CELL_PX * 13, _CELL_PX * 6, 3), dtype=np.uint8)
    )
    rects_all = np.asarray(
        [
            [c * _CELL_PX, r * _CELL_PX, (c + 1) * _CELL_PX, (r + 1) * _CELL_PX]
            for r in range(13)
            for c in range(6)
        ],
        dtype=np.int32,
    )
    rect_one = [
        np.asarray([[0, 0, p.shape[1], p.shape[0]]], dtype=np.int32) for p in patches
    ]

    def a_python() -> None:
        for p in patches:
            clf.classify(p)

    def b_rust_percall() -> None:
        for p, r in zip(patches, rect_one):
            hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
            puyo_core.classify_cells_hsv(p, hsv, r, ranges, params)

    def c_rust_whole() -> None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        puyo_core.classify_cells_hsv(roi, hsv, rects_all, ranges, params)

    print(f"=== 1盤面 {_CELLS}セル / セル {_CELL_PX}x{_CELL_PX}px "
          f"(5回測定の中央値) ===\n")
    print(f"{'方式':<34} {'ms/盤面':>10} {'us/セル':>9} {'対Python':>9}")
    print("-" * 66)
    base = None
    for name, fn in (
        ("A. Python 現行", a_python),
        ("B. Rust パッチ毎 (cvtColor込)", b_rust_percall),
        ("C. Rust 盤面まるごと", c_rust_whole),
    ):
        ts: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(50):
                fn()
            ts.append((time.perf_counter() - t0) / 50 * 1000.0)
        m = statistics.median(ts)
        if base is None:
            base = m
        print(f"{name:<34} {m:9.3f} {m*1000/_CELLS:8.1f} {base/max(m,1e-9):8.1f}x")
    print("-" * 66)
    print("  判断: B が十分速ければ read_board の既存構造をそのまま使える")
    print("        (最小変更)。C が大きく勝つなら ROI+rects を組み直す配線が要る。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
