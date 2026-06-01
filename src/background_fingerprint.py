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

案 d (2026-05-28): PatchBackgroundFingerprint によるパッチ NCC 比較。
    各セル中央 60% パッチを画素単位 float32 HSV で保存し、
    Normalized Cross-Correlation (NCC) で背景類似度を判定。
    単色 HSV 3 数値では表現しきれない複雑なキャラ立ち絵背景に対応。

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

# ===== 案 d: パッチ NCC 閾値 =====
# NCC (Normalized Cross-Correlation) がこれ以上なら「空 (背景と同じ)」と判定。
# 1.0 = 完全一致、0.0 = 無相関、-1.0 = 逆相関。
# 0.92 = 背景との高相関、ぷよ存在時は 0.7〜0.85 程度まで落ちる見込み。
PATCH_NCC_EMPTY_THRESHOLD: float = 0.92
# 均一パッチ (std < この値) で NCC が NaN になる場合のフォールバック NCC 値。
# 均一 → 背景と同じ可能性が高いため 1.0 (= 空判定) を返す。
PATCH_NCC_UNIFORM_FALLBACK: float = 1.0
# std の最小値ガード: これ未満なら均一パッチとみなして NCC 計算をスキップ
PATCH_NCC_STD_MIN: float = 1e-6

# bg_fp 採取失敗パッチの検出閾値 (V channel median)。
# bg_fp 採取時に画面外 / 黒フレーム等で V≈0 のゼロパッチが記録されることがある。
# このようなパッチは「採取失敗」であり、どんな現フレームとも NCC≈1.0 になって
# EMPTY 強制誤判定を引き起こす真因。
# bg パッチの V channel median がこの値未満なら「採取失敗ゼロパッチ」とみなし、
# NCC を 0.0 (= 判定不能 = 非 EMPTY) に倒す。
# 正当な均一空セル (明るい平坦背景) は V median が 20〜255 程度なので影響なし。
BG_PATCH_VALID_V_MIN: float = 5.0


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


def _is_bg_patch_valid(b: np.ndarray) -> bool:
    """bg パッチが採取失敗 (= ゼロパッチ) でないかを確認する。

    bg_fp 採取時に黒フレーム / 画面外で V≈0 のパッチが記録される場合がある。
    そのパッチは std≈0 となり _compute_ncc が PATCH_NCC_UNIFORM_FALLBACK=1.0 を返し、
    どんな現フレームとも「背景と同じ = EMPTY」と誤判定する真因になる。

    b の V channel (index 2) の median が BG_PATCH_VALID_V_MIN 以上であれば有効。
    形状が 1D (ravel 済み) の場合は全体 median で代替する。

    Args:
        b: bg パッチ。shape (H, W, 3) または 1D。

    Returns:
        bool: True = 有効な bg パッチ、False = 採取失敗ゼロパッチ。
    """
    b_arr = np.asarray(b, dtype=np.float64)
    if b_arr.ndim == 3 and b_arr.shape[2] == 3:
        # V channel (index 2) の median で判定
        v_med = float(np.median(b_arr[:, :, 2]))
    else:
        # ravel 済みまたは shape 不明: 全体 median で代替
        v_med = float(np.median(b_arr))
    return v_med >= BG_PATCH_VALID_V_MIN


def _compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのパッチ配列の Normalized Cross-Correlation を計算。

    ゼロ除算ガード付き (std < PATCH_NCC_STD_MIN の場合は均一パッチとして
    PATCH_NCC_UNIFORM_FALLBACK を返す)。

    多層防御: b (= bg パッチ) が採取失敗ゼロパッチ (V median < BG_PATCH_VALID_V_MIN)
    の場合は 0.0 (= 判定不能 = 非 EMPTY) を返す。これにより FALLBACK=1.0 の誤発火を防ぐ。
    a (= 現フレームパッチ) が均一かつ b が有効な場合は FALLBACK を返す (従来通り)。

    Args:
        a: 比較元パッチ (現フレーム、ravel して 1D で計算)
        b: 比較先パッチ (bg FP、同 shape 前提)

    Returns:
        float: NCC 値 (-1.0〜1.0)。採取失敗 bg パッチは 0.0、均一パッチは FALLBACK。
    """
    # 多層防御: bg パッチが採取失敗ゼロパッチなら 0.0 (= 非 EMPTY 側)
    if not _is_bg_patch_valid(b):
        return 0.0
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    if a_flat.std() < PATCH_NCC_STD_MIN or b_flat.std() < PATCH_NCC_STD_MIN:
        # 均一パッチは背景との相関不定 → 背景と同一とみなして高値返却
        return PATCH_NCC_UNIFORM_FALLBACK
    corr = np.corrcoef(a_flat, b_flat)
    ncc = float(corr[0, 1])
    # NaN ガード (万が一 corrcoef が NaN を返した場合)
    return ncc if not np.isnan(ncc) else PATCH_NCC_UNIFORM_FALLBACK


@dataclass(frozen=False)
class CellPatchFingerprint:
    """1 セル分の HSV パッチ (案 d: NCC 比較用)。

    numpy ndarray を保持するため frozen=False。
    外部から patch_hsv を変更しないこと (read-only 規約)。
    """
    # shape: (patch_h, patch_w, 3), dtype: float32, 値域: H=0-180 / S,V=0-255
    patch_hsv: np.ndarray

    def ncc_to(self, other: "CellPatchFingerprint") -> float:
        """現在パッチと other パッチの NCC を返す。

        shape が一致しない場合は resize してから計算。
        """
        a = self.patch_hsv
        b = other.patch_hsv
        # shape 不一致: b を a にリサイズ
        if a.shape != b.shape:
            b = cv2.resize(
                b.astype(np.float32),
                (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return _compute_ncc(a, b)

    def is_empty_by_ncc(
        self,
        bg_patch: "CellPatchFingerprint",
        threshold: float = PATCH_NCC_EMPTY_THRESHOLD,
    ) -> bool:
        """背景パッチとの NCC が threshold 以上なら「空 (背景と同じ)」と判定。"""
        return self.ncc_to(bg_patch) >= threshold


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


def _extract_cell_patch_hsv(
    hsv: np.ndarray,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    row: int,
    col: int,
    sample_ratio: float,
) -> np.ndarray:
    """指定セルの中央パッチ HSV を float32 で切り出す。

    patch が空の場合は zeros(1,1,3) を返す。
    """
    cell_w = region_w / BOARD_COLS
    cell_h = region_h / VISIBLE_ROWS
    cx = int(region_x + (col + 0.5) * cell_w)
    cy = int(region_y + (row + 0.5) * cell_h)
    half_w = max(1, int(cell_w * sample_ratio / 2))
    half_h = max(1, int(cell_h * sample_ratio / 2))
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(hsv.shape[1], cx + half_w)
    y2 = min(hsv.shape[0], cy + half_h)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((1, 1, 3), dtype=np.float32)
    return patch.astype(np.float32)


@dataclass(frozen=True)
class PatchBackgroundFingerprint:
    """6×12 セル分のパッチ画素 FP (案 d: NCC 比較用) + 後退互換 median cells。

    frozen=True だが内部の CellPatchFingerprint.patch_hsv は ndarray のため
    hash() は非対応 (通常は eq 比較のみ使用)。
    """
    # [row][col] の CellPatchFingerprint
    patch_cells: tuple[tuple[CellPatchFingerprint, ...], ...]
    # 後退互換: [row][col] の CellFingerprint (median 3 値)
    median_cells: tuple[tuple[CellFingerprint, ...], ...]

    def cell_at_patch(
        self, row_visible: int, col: int,
    ) -> CellPatchFingerprint:
        """パッチ FP をセル位置で取得。"""
        if 0 <= row_visible < VISIBLE_ROWS and 0 <= col < BOARD_COLS:
            return self.patch_cells[row_visible][col]
        return CellPatchFingerprint(patch_hsv=np.zeros((1, 1, 3), dtype=np.float32))

    def cell_at(self, row_visible: int, col: int) -> CellFingerprint:
        """後退互換: CellFingerprint を返す (BackgroundFingerprint.cell_at と同一 API)。"""
        if 0 <= row_visible < VISIBLE_ROWS and 0 <= col < BOARD_COLS:
            return self.median_cells[row_visible][col]
        return CellFingerprint(0, 0, 0)

    @classmethod
    def capture(
        cls,
        frame: np.ndarray,
        region_x: int,
        region_y: int,
        region_w: int,
        region_h: int,
        sample_ratio: float = CELL_SAMPLE_RATIO,
    ) -> "PatchBackgroundFingerprint":
        """フレームから各セルのパッチ HSV (float32) と median FP を同時採取。

        BackgroundFingerprint.capture と同一シグネチャ (後退互換)。
        """
        if frame is None or frame.ndim != 3:
            empty_patch = tuple(
                tuple(
                    CellPatchFingerprint(
                        patch_hsv=np.zeros((1, 1, 3), dtype=np.float32),
                    )
                    for _ in range(BOARD_COLS)
                )
                for _ in range(VISIBLE_ROWS)
            )
            empty_med = tuple(
                tuple(CellFingerprint(0, 0, 0) for _ in range(BOARD_COLS))
                for _ in range(VISIBLE_ROWS)
            )
            return cls(patch_cells=empty_patch, median_cells=empty_med)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        patch_rows: list[list[CellPatchFingerprint]] = []
        med_rows: list[list[CellFingerprint]] = []
        for r in range(VISIBLE_ROWS):
            pr: list[CellPatchFingerprint] = []
            mr: list[CellFingerprint] = []
            for c in range(BOARD_COLS):
                p = _extract_cell_patch_hsv(
                    hsv, region_x, region_y, region_w, region_h,
                    r, c, sample_ratio,
                )
                pr.append(CellPatchFingerprint(patch_hsv=p))
                mr.append(CellFingerprint(
                    h=int(np.median(p[:, :, 0])),
                    s=int(np.median(p[:, :, 1])),
                    v=int(np.median(p[:, :, 2])),
                ))
            patch_rows.append(pr)
            med_rows.append(mr)
        return cls(
            patch_cells=tuple(tuple(r) for r in patch_rows),
            median_cells=tuple(tuple(r) for r in med_rows),
        )


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


# ===== 案 d: パッチ NCC 関連関数群 =====


def capture_patch_robust_fingerprint(
    frames: "list[np.ndarray]",
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    sample_ratio: float = CELL_SAMPLE_RATIO,
) -> "PatchBackgroundFingerprint":
    """複数フレームのパッチを median 集約してロバスト PatchBackgroundFingerprint を返す。

    各セルごとに shape (n_frames, patch_h, patch_w, 3) バッファを作り
    フレーム軸 (axis=0) で median を取る。
    patch_h/patch_w はフレームごとに統一される前提 (同一 region/ratio)。
    """
    if not frames:
        return PatchBackgroundFingerprint.capture(
            None, region_x, region_y, region_w, region_h, sample_ratio,  # type: ignore[arg-type]
        )
    # 代表フレームで patch サイズを確定
    cell_w = region_w / BOARD_COLS
    cell_h = region_h / VISIBLE_ROWS
    half_w = max(1, int(cell_w * sample_ratio / 2))
    half_h = max(1, int(cell_h * sample_ratio / 2))
    patch_h = half_h * 2
    patch_w = half_w * 2
    # (VISIBLE_ROWS, BOARD_COLS) × n_frames × (patch_h, patch_w, 3)
    n_frames = len(frames)
    buf = np.zeros(
        (VISIBLE_ROWS, BOARD_COLS, n_frames, patch_h, patch_w, 3),
        dtype=np.float32,
    )
    for f_idx, frame in enumerate(frames):
        if frame is None or frame.ndim != 3:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for r in range(VISIBLE_ROWS):
            for c in range(BOARD_COLS):
                p = _extract_cell_patch_hsv(
                    hsv, region_x, region_y, region_w, region_h,
                    r, c, sample_ratio,
                )
                # リサイズで patch_h × patch_w に統一
                if p.shape[:2] != (patch_h, patch_w):
                    p = cv2.resize(p, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
                buf[r, c, f_idx] = p
    # フレーム軸で median (外れ値耐性)
    median_patches = np.median(buf, axis=2)  # (VISIBLE_ROWS, BOARD_COLS, patch_h, patch_w, 3)
    patch_rows: list[list[CellPatchFingerprint]] = []
    med_rows: list[list[CellFingerprint]] = []
    for r in range(VISIBLE_ROWS):
        pr: list[CellPatchFingerprint] = []
        mr: list[CellFingerprint] = []
        for c in range(BOARD_COLS):
            p = median_patches[r, c].astype(np.float32)
            pr.append(CellPatchFingerprint(patch_hsv=p))
            mr.append(CellFingerprint(
                h=int(np.median(p[:, :, 0])),
                s=int(np.median(p[:, :, 1])),
                v=int(np.median(p[:, :, 2])),
            ))
        patch_rows.append(pr)
        med_rows.append(mr)
    return PatchBackgroundFingerprint(
        patch_cells=tuple(tuple(r) for r in patch_rows),
        median_cells=tuple(tuple(r) for r in med_rows),
    )


def capture_patch_pair_robust(
    frames: "list[np.ndarray]",
    p1_region: tuple[int, int, int, int],
    p2_region: tuple[int, int, int, int],
) -> "tuple[PatchBackgroundFingerprint, PatchBackgroundFingerprint]":
    """複数フレームから 1P/2P 両方のロバスト PatchBackgroundFingerprint を取得。"""
    fp1 = capture_patch_robust_fingerprint(frames, *p1_region)
    fp2 = capture_patch_robust_fingerprint(frames, *p2_region)
    return fp1, fp2


def is_empty_by_patch_fp(
    cell_patch: "CellPatchFingerprint",
    bg_patch: "CellPatchFingerprint",
    threshold: float = PATCH_NCC_EMPTY_THRESHOLD,
) -> bool:
    """セルパッチと背景パッチの NCC が threshold 以上なら「空」と判定。"""
    return cell_patch.is_empty_by_ncc(bg_patch, threshold)


# ===== 案 P2: 白ハイライト blob 検出 =====
# ぷよにはゲームエンジン内部で描画された白ハイライト円がセル上部に固定位置で存在する。
# これを検出することで、背景 NCC/距離で空判定された「本物ぷよ」を救済する。

# ハイライト blob 検出閾値: V (輝度) の下限
HIGHLIGHT_V_MIN: int = 220
# ハイライト blob 検出閾値: S (彩度) の上限 (白 = 低彩度)
HIGHLIGHT_S_MAX: int = 50
# 検索領域の上端比率 (セル高さ × ratio が検索開始 y)
HIGHLIGHT_REGION_Y_RATIO: float = 0.15
# 検索領域の下端比率 (セル高さ × ratio が検索終了 y)
HIGHLIGHT_REGION_Y_RATIO_LOW: float = 0.55
# ハイライト blob と判定する最小ピクセル比率 (セル全面積に対する割合)
HIGHLIGHT_MIN_PIXEL_RATIO: float = 0.04


def detect_highlight_blob(
    patch_hsv: np.ndarray,
    v_min: int = HIGHLIGHT_V_MIN,
    s_max: int = HIGHLIGHT_S_MAX,
    y_ratio_top: float = HIGHLIGHT_REGION_Y_RATIO,
    y_ratio_bottom: float = HIGHLIGHT_REGION_Y_RATIO_LOW,
    min_pixel_ratio: float = HIGHLIGHT_MIN_PIXEL_RATIO,
) -> bool:
    """セルパッチ上部帯域で白ハイライト blob を検出する。

    ぷよにはゲームエンジン固定の白ハイライト円が上部に存在する。
    V >= v_min かつ S <= s_max のピクセルがセル全面積の min_pixel_ratio 以上で True。

    Args:
        patch_hsv: セルパッチ HSV。shape (H, W, 3)、float32 or uint8 両対応。
        v_min: 輝度の下限閾値 (デフォルト: HIGHLIGHT_V_MIN)。
        s_max: 彩度の上限閾値 (デフォルト: HIGHLIGHT_S_MAX)。
        y_ratio_top: 検索開始 y 比率 (デフォルト: HIGHLIGHT_REGION_Y_RATIO)。
        y_ratio_bottom: 検索終了 y 比率 (デフォルト: HIGHLIGHT_REGION_Y_RATIO_LOW)。
        min_pixel_ratio: 判定に必要な最小ピクセル比率 (デフォルト: HIGHLIGHT_MIN_PIXEL_RATIO)。

    Returns:
        bool: 白ハイライト blob が検出されれば True。
    """
    if patch_hsv is None or patch_hsv.size == 0:
        return False
    h_size, w_size = patch_hsv.shape[:2]
    # セル全面積 (分母)
    total_pixels = h_size * w_size
    if total_pixels == 0:
        return False
    # 検索帯域: セル上部 y_ratio_top 〜 y_ratio_bottom
    y_start = int(h_size * y_ratio_top)
    y_end = int(h_size * y_ratio_bottom)
    if y_start >= y_end:
        return False
    band = patch_hsv[y_start:y_end, :, :]
    # float32 / uint8 両対応: S, V チャネルを float64 に変換して比較
    s_ch = band[:, :, 1].astype(np.float64)
    v_ch = band[:, :, 2].astype(np.float64)
    # 白マスク: V >= v_min かつ S <= s_max
    white_mask = (v_ch >= float(v_min)) & (s_ch <= float(s_max))
    white_count = int(np.count_nonzero(white_mask))
    return (white_count / total_pixels) >= min_pixel_ratio


def save_patch_fingerprint_pair(
    path: Path,
    fp1: "PatchBackgroundFingerprint",
    fp2: "PatchBackgroundFingerprint",
) -> None:
    """1P/2P PatchBackgroundFingerprint を npz 形式で保存。

    ディレクトリは自動作成する。
    保存形式:
        fp1_patch_{r}_{c}: patch_hsv (float32)
        fp1_med_{r}_{c}_h/s/v: median 値 (int)
        fp2_* 同上
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for side, fp in (("fp1", fp1), ("fp2", fp2)):
        for r in range(VISIBLE_ROWS):
            for c in range(BOARD_COLS):
                arrays[f"{side}_patch_{r}_{c}"] = fp.patch_cells[r][c].patch_hsv
                med = fp.median_cells[r][c]
                arrays[f"{side}_med_{r}_{c}"] = np.array([med.h, med.s, med.v], dtype=np.int32)
    np.savez_compressed(str(path), **arrays)


def load_patch_fingerprint_pair(
    path: Path,
) -> "tuple[PatchBackgroundFingerprint, PatchBackgroundFingerprint]":
    """npz から 1P/2P PatchBackgroundFingerprint をロード。"""
    path = Path(path)
    data = np.load(str(path))
    result = []
    for side in ("fp1", "fp2"):
        patch_rows: list[list[CellPatchFingerprint]] = []
        med_rows: list[list[CellFingerprint]] = []
        for r in range(VISIBLE_ROWS):
            pr: list[CellPatchFingerprint] = []
            mr: list[CellFingerprint] = []
            for c in range(BOARD_COLS):
                p = data[f"{side}_patch_{r}_{c}"].astype(np.float32)
                pr.append(CellPatchFingerprint(patch_hsv=p))
                med_arr = data[f"{side}_med_{r}_{c}"]
                mr.append(CellFingerprint(
                    h=int(med_arr[0]), s=int(med_arr[1]), v=int(med_arr[2]),
                ))
            patch_rows.append(pr)
            med_rows.append(mr)
        result.append(PatchBackgroundFingerprint(
            patch_cells=tuple(tuple(r) for r in patch_rows),
            median_cells=tuple(tuple(r) for r in med_rows),
        ))
    return result[0], result[1]


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
    # 案 d: パッチ NCC 追加
    "PATCH_NCC_EMPTY_THRESHOLD",
    "PATCH_NCC_UNIFORM_FALLBACK",
    "PATCH_NCC_STD_MIN",
    "BG_PATCH_VALID_V_MIN",
    "_is_bg_patch_valid",
    "CellPatchFingerprint",
    "PatchBackgroundFingerprint",
    "capture_patch_robust_fingerprint",
    "capture_patch_pair_robust",
    "is_empty_by_patch_fp",
    "save_patch_fingerprint_pair",
    "load_patch_fingerprint_pair",
    # 案 P2: 白ハイライト blob 検出
    "HIGHLIGHT_V_MIN",
    "HIGHLIGHT_S_MAX",
    "HIGHLIGHT_REGION_Y_RATIO",
    "HIGHLIGHT_REGION_Y_RATIO_LOW",
    "HIGHLIGHT_MIN_PIXEL_RATIO",
    "detect_highlight_blob",
    # T4: 静的背景マスク (pixel-level diff)
    "STATIC_BG_DIFF_THRESHOLD",
    "STATIC_BG_MIN_FRAMES",
    "StaticBoardMask",
    "capture_static_mask",
    "capture_static_mask_pair",
    "save_static_mask_pair",
    "load_static_mask_pair",
]


# ===== T4: 静的背景マスク =====
# 試合開始時の空盤面画像を複数フレームの median で保存し、
# pixel-level BGR max diff で「背景かぷよか」を判定する。
# AND ガード: 色あり signal (HSV hit) があれば EMPTY 化しない。

# per-cell BGR max diff の空判定閾値 (= これ未満なら「背景と同じ」)
STATIC_BG_DIFF_THRESHOLD: float = 30.0
# 安定キャプチャに必要な最小フレーム数
STATIC_BG_MIN_FRAMES: int = 3


@dataclass(frozen=False)
class StaticBoardMask:
    """試合開始時の空盤面 BGR 画像 + per-cell diff 計算機。

    numpy ndarray を保持するため frozen=False。
    bg_roi を外部から書き換えないこと (read-only 規約)。

    Attributes:
        bg_roi: BGR 背景画像 (shape: H×W×3, dtype uint8)
        region_x: 盤面左端 x 座標 (px)
        region_y: 盤面上端 y 座標 (px)
        region_w: 盤面幅 (px)
        region_h: 盤面高さ (px)
    """

    bg_roi: np.ndarray  # shape (H, W, 3), dtype uint8, BGR
    region_x: int
    region_y: int
    region_w: int
    region_h: int

    def _cell_bgr_patch(
        self, frame: np.ndarray, visible_row: int, col: int,
        sample_ratio: float = CELL_SAMPLE_RATIO,
    ) -> tuple[np.ndarray, np.ndarray]:
        """現フレームと背景の per-cell BGR パッチを返す。"""
        cell_w = self.region_w / BOARD_COLS
        cell_h = self.region_h / VISIBLE_ROWS
        cx = int(self.region_x + (col + 0.5) * cell_w)
        cy = int(self.region_y + (visible_row + 0.5) * cell_h)
        half_w = max(1, int(cell_w * sample_ratio / 2))
        half_h = max(1, int(cell_h * sample_ratio / 2))
        img_h, img_w = frame.shape[:2]
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(img_w, cx + half_w)
        y2 = min(img_h, cy + half_h)
        cur_patch = frame[y1:y2, x1:x2].astype(np.float32)
        bg_patch = self.bg_roi[y1:y2, x1:x2].astype(np.float32)
        # shape 不一致ガード (リサイズ)
        if cur_patch.shape != bg_patch.shape and bg_patch.size > 0:
            bg_patch = cv2.resize(
                bg_patch, (cur_patch.shape[1], cur_patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return cur_patch, bg_patch

    def cell_diff_scores(self, frame: np.ndarray) -> np.ndarray:
        """全 visible cell の BGR max diff スコアを計算する。

        Returns:
            shape (VISIBLE_ROWS, BOARD_COLS) の float32 配列。
            値が小さいほど「背景に近い」。
        """
        result = np.zeros((VISIBLE_ROWS, BOARD_COLS), dtype=np.float32)
        for r in range(VISIBLE_ROWS):
            for c in range(BOARD_COLS):
                cur, bg = self._cell_bgr_patch(frame, r, c)
                if cur.size == 0 or bg.size == 0:
                    result[r, c] = 0.0
                    continue
                diff = np.abs(cur - bg)
                result[r, c] = float(np.max(diff))
        return result

    def classify_background_cells(
        self,
        frame: np.ndarray,
        threshold: float = STATIC_BG_DIFF_THRESHOLD,
    ) -> np.ndarray:
        """各 visible cell が「背景と同じ (True)」かを返す。

        Args:
            frame: 現在フレーム (BGR, uint8)
            threshold: max diff 閾値。未満なら True (= 背景)。

        Returns:
            shape (VISIBLE_ROWS, BOARD_COLS) の bool 配列。
        """
        scores = self.cell_diff_scores(frame)
        return scores < threshold


def capture_static_mask(
    frames: "list[np.ndarray]",
    region: "tuple[int, int, int, int]",
) -> "StaticBoardMask":
    """複数フレームの BGR median から StaticBoardMask を生成する。

    Args:
        frames: 試合開始時の空盤面フレーム (BGR, uint8)。
            median を取るため少数のぷよ含みフレームにも頑健。
        region: (x, y, w, h) の盤面領域。

    Returns:
        StaticBoardMask: 生成されたマスク。

    Raises:
        ValueError: len(frames) < STATIC_BG_MIN_FRAMES の場合。
    """
    if len(frames) < STATIC_BG_MIN_FRAMES:
        raise ValueError(
            f"frames 不足: {len(frames)} < {STATIC_BG_MIN_FRAMES}",
        )
    region_x, region_y, region_w, region_h = region
    # フレームスタックで median (外れ値耐性)
    stacked = np.stack(
        [f.astype(np.float32) for f in frames if f is not None and f.ndim == 3],
        axis=0,
    )
    if stacked.shape[0] < STATIC_BG_MIN_FRAMES:
        raise ValueError(
            f"有効フレーム不足: {stacked.shape[0]} < {STATIC_BG_MIN_FRAMES}",
        )
    bg_image = np.median(stacked, axis=0).astype(np.uint8)
    return StaticBoardMask(
        bg_roi=bg_image,
        region_x=region_x,
        region_y=region_y,
        region_w=region_w,
        region_h=region_h,
    )


def capture_static_mask_pair(
    frames: "list[np.ndarray]",
    p1_region: "tuple[int, int, int, int]",
    p2_region: "tuple[int, int, int, int]",
) -> "tuple[StaticBoardMask, StaticBoardMask]":
    """1P/2P 両方の StaticBoardMask を一括生成する。"""
    mask1 = capture_static_mask(frames, p1_region)
    mask2 = capture_static_mask(frames, p2_region)
    return mask1, mask2


def save_static_mask_pair(
    path: "Path",
    mask1: "StaticBoardMask",
    mask2: "StaticBoardMask",
) -> None:
    """1P/2P StaticBoardMask を npz 形式で保存する。

    保存先ディレクトリは自動作成される。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        bg1=mask1.bg_roi,
        bg2=mask2.bg_roi,
        meta1=np.array(
            [mask1.region_x, mask1.region_y, mask1.region_w, mask1.region_h],
            dtype=np.int32,
        ),
        meta2=np.array(
            [mask2.region_x, mask2.region_y, mask2.region_w, mask2.region_h],
            dtype=np.int32,
        ),
    )


def load_static_mask_pair(
    path: "Path",
) -> "tuple[StaticBoardMask, StaticBoardMask]":
    """npz から 1P/2P StaticBoardMask をロードする。"""
    path = Path(path)
    data = np.load(str(path))
    m1 = data["meta1"]
    m2 = data["meta2"]
    mask1 = StaticBoardMask(
        bg_roi=data["bg1"],
        region_x=int(m1[0]), region_y=int(m1[1]),
        region_w=int(m1[2]), region_h=int(m1[3]),
    )
    mask2 = StaticBoardMask(
        bg_roi=data["bg2"],
        region_x=int(m2[0]), region_y=int(m2[1]),
        region_w=int(m2[2]), region_h=int(m2[3]),
    )
    return mask1, mask2
