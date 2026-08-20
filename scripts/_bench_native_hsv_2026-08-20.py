"""Rust の HSV セル分類の速度を Python 実装と比較する (2026-08-20)。

パリティ (bit-identical) は `_verify_native_hsv_parity_2026-08-20.py` で確認済み。
本スクリプトは**効果**を測る。リスクと効果が見合わなければ先送りする判断材料。

比較する2形態:
  A. Python: セルごとに `ColorClassifier.classify(patch)` を呼ぶ (現行)
  B. Rust  : セル矩形をまとめて `classify_cells_hsv` を1回呼ぶ (提案)

B は「呼び出し回数そのもの」を削るので、per-call オーバーヘッドの削減分も
効果に含まれる。これが numpy でのバッチ化 (3回連続で不正解だった) と
本質的に違う点。

計測は同一条件を3回繰り返して中央値を採る (memory: 速度差は単一試行で
判定してはならない)。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

import importlib.util  # noqa: E402

from src.image_reader import ColorClassifier  # noqa: E402

# パリティ検証スクリプトのヘルパーを再利用する (モジュール名にハイフンを
# 含むため通常の import 文が書けない)。パラメータ束と色レンジの平坦化を
# 二重定義しないことが目的 — 定義がずれると測る対象がずれる。
_spec = importlib.util.spec_from_file_location(
    "_parity_helpers",
    PROJECT_ROOT / "scripts" / "_verify_native_hsv_parity_2026-08-20.py",
)
_ph = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ph)
_params = _ph._params
_ranges_flat = _ph._ranges_flat


def _make_roi(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """盤面 ROI 相当のランダム画像 (実画像でなく速度のみを測るため)。"""
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _grid_rects(h: int, w: int, rows: int, cols: int) -> np.ndarray:
    rects = []
    for r in range(rows):
        for c in range(cols):
            rects.append([w * c // cols, h * r // rows,
                          w * (c + 1) // cols, h * (r + 1) // rows])
    return np.asarray(rects, dtype=np.int32)


def main() -> int:
    """1 盤面 (13x6=78セル) の分類にかかる時間を A/B で比較する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cell", type=int, default=16, help="1セルの一辺(px)")
    args = ap.parse_args()

    import puyo_core

    rng = np.random.default_rng(20260820)
    rows, cols = 13, 6
    h, w = args.cell * rows, args.cell * cols
    clf = ColorClassifier()
    ranges = _ranges_flat(clf)
    params = _params(clf)
    rects = _grid_rects(h, w, rows, cols)

    rois = [_make_roi(rng, h, w) for _ in range(20)]
    hsvs = [cv2.cvtColor(r, cv2.COLOR_BGR2HSV) for r in rois]

    print(f"=== 1盤面 {rows}x{cols}={rows*cols}セル / セル {args.cell}x{args.cell}px ===")
    print(f"    {args.iters} 盤面ぶんを {args.repeat} 回測定して中央値を採る\n")

    py_times: list[float] = []
    rs_times: list[float] = []
    for _ in range(args.repeat):
        # A. Python: セルごとに classify
        t0 = time.perf_counter()
        for i in range(args.iters):
            roi = rois[i % len(rois)]
            for x1, y1, x2, y2 in rects:
                clf.classify(roi[y1:y2, x1:x2])
        py_times.append((time.perf_counter() - t0) / args.iters * 1000.0)

        # B. Rust: 盤面まるごと1回
        t0 = time.perf_counter()
        for i in range(args.iters):
            puyo_core.classify_cells_hsv(
                rois[i % len(rois)], hsvs[i % len(hsvs)], rects, ranges, params,
            )
        rs_times.append((time.perf_counter() - t0) / args.iters * 1000.0)

    py = statistics.median(py_times)
    rs = statistics.median(rs_times)
    print(f"{'方式':<34} {'ms/盤面':>10} {'us/セル':>10}")
    print("-" * 56)
    print(f"{'A. Python (セルごとclassify)':<34} {py:9.3f} {py*1000/(rows*cols):9.1f}")
    print(f"{'B. Rust (盤面まるごと1回)':<34} {rs:9.3f} {rs*1000/(rows*cols):9.1f}")
    print("-" * 56)
    print(f"  高速化: {py/max(rs,1e-9):.1f} 倍  (削減 {py-rs:.3f} ms/盤面)")
    print()
    print("  ※ 認識は 1 frame に盤面 2 枚を処理する。")
    print(f"     1 frame あたりの削減見込み: {(py-rs)*2:.2f} ms")
    print("     実測の classify 起因コストは 19.64ms/frame (中盤、中央値88回)。")
    print("     ただし本ベンチはランダム画像なので、実画像での")
    print("     サブパッチ再判定の発生頻度は反映されていない。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
