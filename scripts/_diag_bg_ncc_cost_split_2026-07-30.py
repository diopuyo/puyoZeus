"""背景照合 (is_empty_by_ncc) の 137us/回 の内訳を切り分ける (2026-07-30)。

実測: 16.4ms/frame、120回/frame = 1回あたり約137us で認識全体の15.2%。
どこが重いのか (どれを直せば効くのか) を推測せず測る。

候補:
  1. image_reader.py:988 の `np.median(bg_cell.patch_hsv[:,:,2])`
     — 背景パッチは不変なのに毎フレーム・全セルで再計算している
  2. `CellPatchFingerprint(patch_hsv=cur.astype(np.float32))` の型変換複製
  3. `ncc_to` 内の `a.ravel().astype(np.float64)` と `.std()`
  4. **`np.corrcoef`** — 2xN 行列を作って 2x2 共分散行列を経由する重い実装

対策候補の比較も行う:
  - 直接ピアソン (dot 3 回) への置換
  - 全セルまとめて行列積 (スコアOCRで146倍速を実証した手法)
"""

from __future__ import annotations

import time

import numpy as np

# セルパッチの想定サイズ (CELL_SAMPLE_RATIO 適用後。HSV 3ch)
PATCH_SHAPES: tuple[tuple[int, int], ...] = ((16, 16), (24, 24), (32, 32), (40, 40))
CELLS_PER_FRAME: int = 120
REPEAT: int = 500


def _bench(fn, *args) -> float:
    """1 回あたりの所要秒。"""
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn(*args)
    return (time.perf_counter() - t0) / REPEAT


def _corrcoef_path(a: np.ndarray, b_flat: np.ndarray) -> float:
    """現行相当: ravel + astype(float64) + std + np.corrcoef。"""
    a_flat = a.ravel().astype(np.float64)
    if a_flat.std() < 1e-6:
        return 1.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _direct_pearson(a: np.ndarray, b_centered: np.ndarray, b_norm: float) -> float:
    """直接ピアソン: 平均引き + dot 2 回 (b 側は前計算済み)。"""
    v = a.ravel().astype(np.float64)
    v = v - v.mean()
    n = float(np.sqrt(np.dot(v, v)))
    if n == 0.0:
        return 1.0
    return float(np.dot(v, b_centered) / (n * b_norm))


def main() -> None:
    rng = np.random.default_rng(42)
    print(f"{'パッチ':>10}{'画素数':>8}{'corrcoef':>12}{'直接ピアソン':>14}{'倍率':>8}")
    for h, w in PATCH_SHAPES:
        a = rng.random((h, w, 3), dtype=np.float64).astype(np.float32) * 255
        b = rng.random((h, w, 3), dtype=np.float64).astype(np.float32) * 255
        b_flat = b.ravel().astype(np.float64)
        b_centered = b_flat - b_flat.mean()
        b_norm = float(np.sqrt(np.dot(b_centered, b_centered)))

        t_cc = _bench(_corrcoef_path, a, b_flat)
        t_dp = _bench(_direct_pearson, a, b_centered, b_norm)
        # 一致度
        d = abs(_corrcoef_path(a, b_flat) - _direct_pearson(a, b_centered, b_norm))
        print(
            f"{h}x{w:<6}{h * w * 3:>8}{t_cc * 1e6:>10.2f}us{t_dp * 1e6:>12.2f}us"
            f"{t_cc / t_dp:>7.1f}倍  (差 {d:.2e})"
        )

    # --- 個別ループ vs 全セルまとめ行列積 ---
    print(f"\n=== {CELLS_PER_FRAME}セル分 (1フレーム相当) ===")
    for h, w in ((24, 24), (32, 32)):
        n_px = h * w * 3
        cells = rng.random((CELLS_PER_FRAME, n_px)) * 255
        bgs = rng.random((CELLS_PER_FRAME, n_px)) * 255
        # 背景側は前計算済み (平均引き + L2 正規化)
        bg_c = bgs - bgs.mean(axis=1, keepdims=True)
        bg_c /= np.linalg.norm(bg_c, axis=1, keepdims=True)

        def _loop(cs: np.ndarray, bc: np.ndarray) -> np.ndarray:
            """セルごとに直接ピアソン。"""
            out = np.empty(cs.shape[0])
            for i in range(cs.shape[0]):
                v = cs[i] - cs[i].mean()
                out[i] = np.dot(v, bc[i]) / np.sqrt(np.dot(v, v))
            return out

        def _batched(cs: np.ndarray, bc: np.ndarray) -> np.ndarray:
            """全セルまとめて: 平均引き → 行ごとの内積を einsum 1 回。"""
            v = cs - cs.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.einsum("ij,ij->i", v, v))
            return np.einsum("ij,ij->i", v, bc) / norms

        t_loop = _bench(_loop, cells, bg_c)
        t_batch = _bench(_batched, cells, bg_c)
        maxd = float(np.abs(_loop(cells, bg_c) - _batched(cells, bg_c)).max())
        print(
            f"  {h}x{w}: 個別ループ {t_loop * 1e3:6.2f}ms  "
            f"まとめ {t_batch * 1e3:6.2f}ms  "
            f"({t_loop / t_batch:.1f}倍速、最大差 {maxd:.2e})"
        )


if __name__ == "__main__":
    main()
