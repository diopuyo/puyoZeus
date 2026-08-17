"""STABLE 確定盤面の生ピクセル持続確認 (収集限定、2026-08-18 新設)。

docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §5 (d) の実装。
`scripts/_diag_general_chain_contamination_2026-08-17.py:160-176` の
`_board_roi_gray`/`_column_diffs` を stateless 純関数として昇格したもの。

対象カテゴリ: ①連鎖アニメ中の STABLE 誤確定 ②送付フラッシュ重畳。
③試合外 (静止画面) は生ピクセル差分が元々ゼロのため本機構では検出できない
(design doc §5、(b) の守備範囲であり (d) の代替にはならない)。

state を一切持たない (直近 diff の履歴保持は呼び出し側 = collect_boards_lean.py
の責務、CLAUDE.md 「観測指標は stateless 実装を原則」に従う)。
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

# 盤面 ROI を等分する列数 (6列盤面、cell 単位の空間分解のため)。
N_BOARD_COLUMNS: int = 6

# STABLE持続確認の diff 閾値 (2026-08-18 実測校正、
# data/verify/diag_general_chain_contamination_2026-08-17/
# classification_corrected_2026-08-17.json より):
# 連鎖/送付フラッシュ混入 (アンカー瞬間の diff_mean、静止画面混入=030を除く
# 6件中5件が対象): 003=4.711, 005=4.196, 031=1.417, 012=1.146(低確信),
# 029=1.07 (最小値)。
# 綺麗な21枚 (final_tally A_clean_static) の min_diff_near 最大値: 0.858
# (009_c13_2P)。0.858 < 1.0 < 1.07 の分離ギャップに収まるラウンド値を採用
# (個別ケースへの逆算ではなく全27件の実測分布から)。
STABLE_PERSISTENCE_DIFF_THRESHOLD: float = 1.0

# STABLE持続確認の窓長 (2026-08-18): 上記 ground truth 構築時に使われた
# ±0.25 秒窓 (_diag_general_chain_contamination_2026-08-17.py の
# DENSE_WINDOW_SEC 由来の分析窓) と同じ根拠を再利用する。
STABLE_PERSISTENCE_WINDOW_SEC: float = 0.25


def board_roi_gray(frame: np.ndarray, side: str) -> np.ndarray:
    """盤面 ROI (P1/P2 領域) を grayscale で切り出す。

    Args:
        frame: 1920x1080 にリサイズ済のフレーム (BGR)。
        side: "1P" または "2P"。

    Returns:
        grayscale の盤面 ROI 画像 (H, W)。
    """
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    crop = frame[
        region.y: region.y + region.height,
        region.x: region.x + region.width,
    ]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def column_diffs(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    n_cols: int = N_BOARD_COLUMNS,
) -> "list[float]":
    """列ごとの平均絶対差分 (盤面幅を n_cols 等分)。

    _diag_general_chain_contamination_2026-08-17.py の `_column_diffs` と
    数値的に同一 (回帰テストで一致確認)。
    """
    _h, w = cur_gray.shape
    col_w = w / n_cols
    diff = np.abs(cur_gray.astype(np.int16) - prev_gray.astype(np.int16))
    out: "list[float]" = []
    for i in range(n_cols):
        x1, x2 = int(i * col_w), int((i + 1) * col_w)
        out.append(float(diff[:, x1:x2].mean()))
    return out


def frame_diff_mean(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    n_cols: int = N_BOARD_COLUMNS,
) -> float:
    """列ごと平均絶対差分 (column_diffs) の全体平均。

    診断データの `diff_mean` (classification_corrected_2026-08-17.json の
    `diff_at_exact_anchor`) と同一の定義。
    """
    diffs = column_diffs(prev_gray, cur_gray, n_cols=n_cols)
    return float(np.mean(diffs)) if diffs else 0.0


def is_raw_pixel_stable(
    recent_diffs: Sequence[float],
    diff_threshold: float = STABLE_PERSISTENCE_DIFF_THRESHOLD,
) -> bool:
    """直近ウィンドウ内の diff が全て閾値未満なら True (持続静止)。

    Args:
        recent_diffs: 直近 STABLE_PERSISTENCE_WINDOW_SEC 秒分の
            frame_diff_mean 値 (呼び出し側が rolling window で保持する)。
            空リスト (= まだ直前フレームが無く diff 計算不能な立ち上がり
            直後) の場合は保守的に安全側 = True を返す (収集を止めない)。
        diff_threshold: 閾値 (既定 STABLE_PERSISTENCE_DIFF_THRESHOLD)。

    Returns:
        recent_diffs が全て diff_threshold 未満なら True。
    """
    if not recent_diffs:
        return True
    return all(d < diff_threshold for d in recent_diffs)
