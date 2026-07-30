"""スコアOCRの同サイズNCCを行列積化した場合の速度と一致度を測る (2026-07-30)。

背景 (実測):
    score_ocr の matchTemplate が認識全体の 19.9%、160回/フレーム、ROI/テンプレとも
    50x40 の同サイズ。同サイズ TM_CCOEFF_NORMED は結果が 1x1 = **Pearson 相関**なので、
    「平均を引いて L2 正規化したテンプレ行列 × セルベクトル」の 1 回の行列積で
    全テンプレ分を同時に得られる。テンプレは固定なので正規化は前計算できる。

測る 3 点:
    1. 現行 (テンプレ数ぶん matchTemplate をループ) の時間
    2. numpy の行列積 1 回の時間
    3. torch CUDA の行列積 1 回の時間 (**転送コストを含めて**)
       → データが小さいので GPU 転送が不利になる可能性を確かめる。
          user 指示の「GPUモード」に見合う標的かどうかの判断材料。

    併せて **スコアの一致度** (最大絶対差) を出す。行列積は演算順序が違うため
    bit-identical にはならない。ラベル決定が変わらないかを件数で確認する。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_score_ocr_matmul_bench_2026-07-30
"""

from __future__ import annotations

import time

import cv2
import numpy as np

# score_ocr の実測 ROI サイズ (DIGIT_HEIGHT x DIGIT_WIDTH)
CELL_H, CELL_W = 50, 40
# 1 フレームあたりの実測呼び出し回数 (= セル数 x テンプレ数)
CALLS_PER_FRAME: int = 160
# ベンチの反復回数
REPEAT: int = 300


def _ncc_loop(cell: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    """現行方式: テンプレごとに matchTemplate を呼ぶ。"""
    return np.array(
        [
            float(cv2.matchTemplate(cell, t, cv2.TM_CCOEFF_NORMED).max())
            for t in templates
        ],
        dtype=np.float64,
    )


def _prepare_template_matrix(templates: list[np.ndarray]) -> np.ndarray:
    """テンプレ行列を「平均引き + L2 正規化」して前計算する (固定なので 1 回だけ)。"""
    mat = np.stack([t.ravel().astype(np.float64) for t in templates])
    mat -= mat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _ncc_matmul(cell: np.ndarray, tpl_mat: np.ndarray) -> np.ndarray:
    """行列積方式: 正規化済みテンプレ行列とセルベクトルの積で全スコアを一括計算。"""
    v = cell.ravel().astype(np.float64)
    v = v - v.mean()
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.zeros(tpl_mat.shape[0], dtype=np.float64)
    return tpl_mat @ (v / n)


def _bench(fn, *args) -> float:
    """1 回あたりの所要秒。"""
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn(*args)
    return (time.perf_counter() - t0) / REPEAT


def main() -> None:
    cv2.setNumThreads(1)
    rng = np.random.default_rng(42)
    n_templates = 10  # 数字 0-9
    templates = [
        rng.integers(0, 256, size=(CELL_H, CELL_W), dtype=np.uint8)
        for _ in range(n_templates)
    ]
    cells = [
        rng.integers(0, 256, size=(CELL_H, CELL_W), dtype=np.uint8)
        for _ in range(16)
    ]
    tpl_mat = _prepare_template_matrix(templates)

    # --- 一致度 ---
    max_abs_diff = 0.0
    label_mismatch = 0
    for cell in cells:
        a = _ncc_loop(cell, templates)
        b = _ncc_matmul(cell, tpl_mat)
        max_abs_diff = max(max_abs_diff, float(np.abs(a - b).max()))
        if int(np.argmax(a)) != int(np.argmax(b)):
            label_mismatch += 1
    print(f"=== 一致度 (16セル x {n_templates}テンプレ) ===")
    print(f"  スコア最大絶対差: {max_abs_diff:.3e}")
    print(f"  ラベル決定の不一致: {label_mismatch}/{len(cells)} セル")

    # --- 速度 (1 セル分 = テンプレ数ぶんのスコア取得) ---
    cell = cells[0]
    t_loop = _bench(_ncc_loop, cell, templates)
    t_mm = _bench(_ncc_matmul, cell, tpl_mat)
    print(f"\n=== 速度 (1セル分 = {n_templates}テンプレ分のスコア) ===")
    print(f"  現行 matchTemplate ループ: {t_loop * 1e6:8.2f}us")
    print(f"  numpy 行列積            : {t_mm * 1e6:8.2f}us  ({t_loop / t_mm:.1f}倍速)")

    # --- torch CUDA (転送込み) ---
    try:
        import torch
    except ImportError:
        print("\n  torch 未インストール: GPU 比較を skip")
        torch = None  # type: ignore[assignment]
    if torch is not None:
        print(f"\n=== torch (CUDA 利用可能: {torch.cuda.is_available()}) ===")
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            tpl_gpu = torch.from_numpy(tpl_mat).to(dev)

            def _ncc_torch_cuda(c: np.ndarray) -> np.ndarray:
                """転送込みの GPU 行列積 (毎フレーム新しいセルを送る前提)。"""
                v = torch.from_numpy(c.ravel().astype(np.float64)).to(dev)
                v = v - v.mean()
                v = v / torch.linalg.norm(v)
                return (tpl_gpu @ v).cpu().numpy()

            _ncc_torch_cuda(cell)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT):
                _ncc_torch_cuda(cell)
            torch.cuda.synchronize()
            t_gpu = (time.perf_counter() - t0) / REPEAT
            print(f"  CUDA 行列積 (転送込み)  : {t_gpu * 1e6:8.2f}us  "
                  f"({t_loop / t_gpu:.1f}倍速 / numpy比 {t_mm / t_gpu:.2f}倍)")
            if t_gpu > t_mm:
                print("  → **GPU は numpy より遅い**。データが小さく転送コストが支配的。")

    # --- 1 フレーム換算 ---
    per_frame_loop = t_loop * (CALLS_PER_FRAME / n_templates)
    per_frame_mm = t_mm * (CALLS_PER_FRAME / n_templates)
    print(f"\n=== 1フレーム換算 ({CALLS_PER_FRAME}回/frame = "
          f"{CALLS_PER_FRAME // n_templates}セル x {n_templates}テンプレ) ===")
    print(f"  現行  : {per_frame_loop * 1e3:6.2f}ms")
    print(f"  行列積: {per_frame_mm * 1e3:6.2f}ms  "
          f"→ 削減 {(per_frame_loop - per_frame_mm) * 1e3:.2f}ms/frame")


if __name__ == "__main__":
    main()
