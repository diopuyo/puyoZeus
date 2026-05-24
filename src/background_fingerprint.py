"""盤面背景フィンガープリント (Phase T サイクル 1)。

ユーザ提案: 試合開始時に「空盤面」の HSV 値を保存し、推論時に背景との差分で
ぷよの存在判定を強化する。

原理:
    試合開始 0.5-2.0 秒頃のフレームでは盤面はまだ空。
    各セルの中央パッチの (H, S, V) 中央値を「背景 FP」として保存。
    推論時に「現在のパッチ」と「背景 FP」の HSV 距離を計算し、
        - 距離 < EMPTY_THRESHOLD → 空 (背景のまま、ぷよなし)
        - 距離 ≥ EMPTY_THRESHOLD → ぷよあり (HSV/CNN で色判定)

利点:
    - キャラクター背景・UI 装飾の影響を打ち消し
    - エフェクトの一部 (背景の発光等) も無視できる
    - 各動画/各試合で動的キャリブレーション

利用例:
    fp = BackgroundFingerprint.capture(frame_at_match_start, region)
    is_empty_grid = fp.classify_empty(frame, region)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.board import BOARD_COLS, HIDDEN_ROWS, VISIBLE_ROWS

# 背景との HSV 距離閾値: これ未満なら「空セル (背景と同じ)」
# H は 0-180、S/V は 0-255 → 重み付け Manhattan 距離で 0-200 程度
# cycle 3 (2026-05-15, F3): 28.0 → 35.0 で aggressive bg match (背景 puyo 誤認↓).
# cycle 17 (2026-05-16, B): 35.0 → 50.0. cnn_phase_i_hsv_seed.pt (cycle_14 model)
# は puyo 認識完璧だが empty 学習なし。 bg_fp 閾値を上げて 1st pass の早期 EMPTY
# 判定を aggressive 化し、 背景 cell を HybridClassifier に渡さない方針。
DEFAULT_EMPTY_HSV_DISTANCE: float = 50.0
# H の重み (S/V より小さく、彩度低い背景でも安定)
H_WEIGHT: float = 0.5
S_WEIGHT: float = 1.0
V_WEIGHT: float = 1.0
# サンプリング比率 (cell width/height のうち中央何 % をサンプル)
CELL_SAMPLE_RATIO: float = 0.6


@dataclass(frozen=True)
class CellFingerprint:
    """1 セル分の HSV 中央値。"""
    h: int
    s: int
    v: int

    def distance_to(self, other: "CellFingerprint") -> float:
        """重み付き HSV 距離。"""
        # H は循環するので最短距離
        dh = abs(int(self.h) - int(other.h))
        dh = min(dh, 180 - dh)
        ds = abs(int(self.s) - int(other.s))
        dv = abs(int(self.v) - int(other.v))
        return H_WEIGHT * dh + S_WEIGHT * ds + V_WEIGHT * dv


@dataclass(frozen=True)
class BackgroundFingerprint:
    """6×12 = 72 セル分の背景 HSV FP。"""
    cells: tuple[tuple[CellFingerprint, ...], ...]  # [row][col]

    @classmethod
    def capture(
        cls,
        frame: np.ndarray,
        region_x: int,
        region_y: int,
        region_w: int,
        region_h: int,
        sample_ratio: float = CELL_SAMPLE_RATIO,
    ) -> "BackgroundFingerprint":
        """フレームから盤面領域の各セル中央 HSV 中央値を取得。"""
        if frame is None or frame.ndim != 3:
            empty = tuple(
                tuple(CellFingerprint(0, 0, 0) for _ in range(BOARD_COLS))
                for _ in range(VISIBLE_ROWS)
            )
            return cls(cells=empty)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cell_w = region_w / BOARD_COLS
        cell_h = region_h / VISIBLE_ROWS
        rows: list[list[CellFingerprint]] = []
        for r in range(VISIBLE_ROWS):
            row_cells: list[CellFingerprint] = []
            for c in range(BOARD_COLS):
                cx = int(region_x + (c + 0.5) * cell_w)
                cy = int(region_y + (r + 0.5) * cell_h)
                half_w = max(1, int(cell_w * sample_ratio / 2))
                half_h = max(1, int(cell_h * sample_ratio / 2))
                x1 = max(0, cx - half_w)
                y1 = max(0, cy - half_h)
                x2 = min(hsv.shape[1], cx + half_w)
                y2 = min(hsv.shape[0], cy + half_h)
                patch = hsv[y1:y2, x1:x2]
                if patch.size == 0:
                    row_cells.append(CellFingerprint(0, 0, 0))
                    continue
                h_med = int(np.median(patch[:, :, 0]))
                s_med = int(np.median(patch[:, :, 1]))
                v_med = int(np.median(patch[:, :, 2]))
                row_cells.append(CellFingerprint(h_med, s_med, v_med))
            rows.append(row_cells)
        return cls(
            cells=tuple(tuple(r) for r in rows),
        )

    def cell_at(self, row_visible: int, col: int) -> CellFingerprint:
        """可視行 (0..11) と列でセル取得。
        Board の row (0..12) を渡す場合は HIDDEN_ROWS を引いてから渡すこと。
        """
        if 0 <= row_visible < VISIBLE_ROWS and 0 <= col < BOARD_COLS:
            return self.cells[row_visible][col]
        return CellFingerprint(0, 0, 0)


def is_empty_by_fp(
    cell_hsv_med: CellFingerprint,
    bg_fp: CellFingerprint,
    threshold: float = DEFAULT_EMPTY_HSV_DISTANCE,
) -> bool:
    """セル中央 HSV と背景 FP の距離が閾値未満なら「空」と判定。"""
    return cell_hsv_med.distance_to(bg_fp) < threshold


def capture_pair_fingerprint(
    frame: np.ndarray,
    p1_region: tuple[int, int, int, int],  # (x, y, w, h)
    p2_region: tuple[int, int, int, int],
) -> tuple[BackgroundFingerprint, BackgroundFingerprint]:
    """1P/2P 両方の背景 FP を一括取得。"""
    fp1 = BackgroundFingerprint.capture(frame, *p1_region)
    fp2 = BackgroundFingerprint.capture(frame, *p2_region)
    return fp1, fp2


def capture_robust_fingerprint(
    frames: "list[np.ndarray]",
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    sample_ratio: float = CELL_SAMPLE_RATIO,
) -> BackgroundFingerprint:
    """複数フレームから各セルの HSV を集約し、外れ値除去した背景 FP を返す。

    試合開始時の単一フレームでは「キャラクター背景の瞬間的な動き」「UI 装飾」
    で背景値が偏る可能性があるため、複数フレームの median を採ってロバスト化。

    アルゴリズム:
        各セル中央パッチで全フレーム分の median HSV を保存 →
        フレーム軸で各 (h, s, v) の median を取って 1 つの CellFingerprint 化。

    Args:
        frames: 試合開始数秒の盤面が空であろうフレーム複数枚。
            空盤面でなくとも、median を取るので少数のぷよ含みフレームには頑健。
        region_x/y/w/h: 盤面領域。
        sample_ratio: セル中央のサンプル比率。

    Returns:
        BackgroundFingerprint: 全セルの HSV median を持つ FP。
    """
    if not frames:
        empty = tuple(
            tuple(CellFingerprint(0, 0, 0) for _ in range(BOARD_COLS))
            for _ in range(VISIBLE_ROWS)
        )
        return BackgroundFingerprint(cells=empty)

    cell_w = region_w / BOARD_COLS
    cell_h = region_h / VISIBLE_ROWS
    # 各セル × 各フレームの (h, s, v) を蓄積
    n_frames = len(frames)
    h_buffer = np.zeros((VISIBLE_ROWS, BOARD_COLS, n_frames), dtype=np.float32)
    s_buffer = np.zeros((VISIBLE_ROWS, BOARD_COLS, n_frames), dtype=np.float32)
    v_buffer = np.zeros((VISIBLE_ROWS, BOARD_COLS, n_frames), dtype=np.float32)
    for f_idx, frame in enumerate(frames):
        if frame is None or frame.ndim != 3:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for r in range(VISIBLE_ROWS):
            for c in range(BOARD_COLS):
                cx = int(region_x + (c + 0.5) * cell_w)
                cy = int(region_y + (r + 0.5) * cell_h)
                half_w = max(1, int(cell_w * sample_ratio / 2))
                half_h = max(1, int(cell_h * sample_ratio / 2))
                x1 = max(0, cx - half_w)
                y1 = max(0, cy - half_h)
                x2 = min(hsv.shape[1], cx + half_w)
                y2 = min(hsv.shape[0], cy + half_h)
                patch = hsv[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                h_buffer[r, c, f_idx] = float(np.median(patch[:, :, 0]))
                s_buffer[r, c, f_idx] = float(np.median(patch[:, :, 1]))
                v_buffer[r, c, f_idx] = float(np.median(patch[:, :, 2]))
    # フレーム軸で median (外れ値耐性)
    rows: list[list[CellFingerprint]] = []
    for r in range(VISIBLE_ROWS):
        row_cells: list[CellFingerprint] = []
        for c in range(BOARD_COLS):
            h_med = int(np.median(h_buffer[r, c]))
            s_med = int(np.median(s_buffer[r, c]))
            v_med = int(np.median(v_buffer[r, c]))
            row_cells.append(CellFingerprint(h_med, s_med, v_med))
        rows.append(row_cells)
    return BackgroundFingerprint(cells=tuple(tuple(r) for r in rows))


def capture_pair_robust(
    frames: "list[np.ndarray]",
    p1_region: tuple[int, int, int, int],
    p2_region: tuple[int, int, int, int],
) -> tuple[BackgroundFingerprint, BackgroundFingerprint]:
    """複数フレームから 1P/2P 両方のロバスト背景 FP を取得 (T-v2-N)。"""
    fp1 = capture_robust_fingerprint(frames, *p1_region)
    fp2 = capture_robust_fingerprint(frames, *p2_region)
    return fp1, fp2


__all__ = [
    "BackgroundFingerprint",
    "CellFingerprint",
    "DEFAULT_EMPTY_HSV_DISTANCE",
    "H_WEIGHT",
    "S_WEIGHT",
    "V_WEIGHT",
    "capture_pair_fingerprint",
    "capture_pair_robust",
    "capture_robust_fingerprint",
    "is_empty_by_fp",
]
