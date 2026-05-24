"""半自動 cell ラベリング tool (cycle 71h+ / case 1+4).

設計:
    動画から N frame をサンプリング、 各 frame で 1P/2P 盤面の visible cells
    (= row 1-12, col 0-5 = 72 cells/盤面 = 144 cells/frame) を grid 表示.
    ColorClassifier 予測で pre-fill し、 ユーザーがクリック + 色キーで
    誤認 cell を修正、 PseudoLabelSample 形式で `data/pseudo_labels/<vid>/cell.jsonl`
    に追記する.

CNN 再訓練 pipeline (= phase_i_fine_tune.py --component cell_color) との互換性:
    - component = "cell"
    - input_data = {"patch": BGR ndarray}
    - label = COLOR_* (int)
    - confidence = 1.0 (manual)
    - metadata = {"video_id", "frame_idx", "row", "col", "side", "manual": True}

操作:
    マウスクリック    : cell 選択
    数値キー 1-5      : 色設定 (1=赤, 2=青, 3=緑, 4=黄, 5=紫)
    数値キー 9        : お邪魔ぷよ
    数値キー 0        : 空 cell
    頭文字 R/Y/P/B/G/O/E : 色設定 (大文字小文字どちらでも)
                          R=赤, Y=黄, P=紫, B=青, G=緑, O=お邪魔, E=空
    Space             : 次 frame (current frame の全 cell を保存)
    [ or ,            : 前 frame (B=BLUE 用に B 移動から変更)
    ] or .            : 次 frame (= Space と同等)
    A                 : 全 cell を CNN 予測のまま承認 (= 一括確定)
    U                 : 最後の操作を undo
    S                 : 即座に保存 (frame 跨ぎでも)
    Q                 : 終了

使い方:
    PYTHONPATH=. python -m scripts.label_cells \
        --video data/test_unknown/v50_match1_75s_720p.mp4 \
        --video-id v50_match1 \
        --start-sec 48 --end-sec 75 \
        --interval-sec 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN,
    COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.image_reader import (
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, ColorClassifier,
)
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL, COMPONENT_FRAME_STATE, PseudoLabelSample,
)


# フレーム単位 state ラベル (= X-2 案、 2026-05-12)
# 2 軸独立設計:
#   軸 1 = 動作状態 (排他、 下記の 6 値から 1 つ)
#   軸 2 = 演出フラグ (= ON/OFF、 動作状態と独立)
# 「CHAIN + 演出あり」 「TSUMO_FALL + 演出あり」 等の両立シナリオに対応.
FRAME_STATE_STABLE: str = "STABLE"
FRAME_STATE_CHAIN: str = "CHAIN"
FRAME_STATE_OJAMA_FALL: str = "OJAMA_FALL"
FRAME_STATE_TSUMO_FALL: str = "TSUMO_FALL"
FRAME_STATE_MENU: str = "MENU"
# Skip = ラベリングしない (= 保存せず飛ばす特殊値)
FRAME_STATE_SKIP: str = "SKIP"
# 旧 EFFECT は 動作状態から削除し、 effect_<side>: bool フラグに分離.
# 後方互換のため値だけ残す (= 既存ラベルデータ読み込み用).
FRAME_STATE_EFFECT: str = "EFFECT"

# キー → 動作状態マッピング (= 1P/2P 別、 排他選択)
KEY_TO_FRAME_STATE: dict[str, str] = {
    "n": FRAME_STATE_STABLE,    # Normal
    "c": FRAME_STATE_CHAIN,
    "j": FRAME_STATE_OJAMA_FALL,  # おじゃま (J: jama)
    "t": FRAME_STATE_TSUMO_FALL,
    "m": FRAME_STATE_MENU,
    "k": FRAME_STATE_SKIP,        # Skip (= ラベリング除外)
}
# 演出フラグ用キー (= 動作状態とは独立、 toggle).
# F = 1P 演出 toggle, Shift+F = 2P 演出 toggle.
KEY_EFFECT_TOGGLE: str = "f"


# ============================
# 表示用定数
# ============================

# 1 cell の grid サムネイルサイズ (= UI 強化で大きく)
CELL_THUMB_W: int = 84
CELL_THUMB_H: int = 84
# 表示行 (= 可視 row のみ、 row 1-12 = 12 行)
VISIBLE_ROW_START: int = HIDDEN_ROWS  # = 1
VISIBLE_ROW_END: int = BOARD_ROWS  # = 13 (exclusive)
VISIBLE_ROW_COUNT: int = VISIBLE_ROW_END - VISIBLE_ROW_START  # = 12

# 拡大プレビュー
PREVIEW_PX: int = 200
# 色パレットボタン
PALETTE_BTN_W: int = 100
PALETTE_BTN_H: int = 60

# UI レイアウト
# 構造: 左 = 1P grid | 中央 = 2P grid | 右 = preview + palette
GRID_W: int = CELL_THUMB_W * BOARD_COLS  # = 504
GRID_H: int = CELL_THUMB_H * VISIBLE_ROW_COUNT  # = 1008
SEP_W: int = 24  # 1P / 2P 間の separator
RIGHT_PANE_W: int = 260
STATUS_H: int = 130
WIN_W: int = GRID_W * 2 + SEP_W + RIGHT_PANE_W
WIN_H: int = GRID_H + STATUS_H

# 色 → BGR (描画用)
COLOR_TO_BGR: dict[int, tuple[int, int, int]] = {
    COLOR_EMPTY: (40, 40, 40),
    COLOR_RED: (0, 0, 255),
    COLOR_BLUE: (255, 0, 0),
    COLOR_GREEN: (0, 200, 0),
    COLOR_YELLOW: (0, 255, 255),
    COLOR_PURPLE: (200, 0, 200),
    COLOR_OJAMA: (140, 140, 140),
    COLOR_UNKNOWN: (255, 255, 255),
}
COLOR_TO_NAME: dict[int, str] = {
    COLOR_EMPTY: "EMP",
    COLOR_RED: "RED",
    COLOR_BLUE: "BLU",
    COLOR_GREEN: "GRN",
    COLOR_YELLOW: "YEL",
    COLOR_PURPLE: "PUR",
    COLOR_OJAMA: "OJM",
    COLOR_UNKNOWN: "UNK",
}
# キー → 色 (数値 + 頭文字 RYPBGOE + UNKNOWN=X)
KEY_TO_COLOR: dict[int, int] = {
    # 数値キー
    ord("1"): COLOR_RED,
    ord("2"): COLOR_BLUE,
    ord("3"): COLOR_GREEN,
    ord("4"): COLOR_YELLOW,
    ord("5"): COLOR_PURPLE,
    ord("9"): COLOR_OJAMA,
    ord("0"): COLOR_EMPTY,
    # 頭文字キー (大文字小文字対応、 ユーザー要望)
    ord("r"): COLOR_RED,    ord("R"): COLOR_RED,
    ord("y"): COLOR_YELLOW, ord("Y"): COLOR_YELLOW,
    ord("p"): COLOR_PURPLE, ord("P"): COLOR_PURPLE,
    ord("b"): COLOR_BLUE,   ord("B"): COLOR_BLUE,
    ord("g"): COLOR_GREEN,  ord("G"): COLOR_GREEN,
    ord("o"): COLOR_OJAMA,  ord("O"): COLOR_OJAMA,
    ord("e"): COLOR_EMPTY,  ord("E"): COLOR_EMPTY,
    # UNKNOWN (= 落下中のぷよ、 色不明な cell)
    ord("x"): COLOR_UNKNOWN, ord("X"): COLOR_UNKNOWN,
}


# ============================
# Frame 単位の cell labels 管理
# ============================


class FrameLabels:
    """1 frame 分の cell label 管理 (1P + 2P) + frame state ラベル.

    X-2 案 (2026-05-12): frame_state でフレーム全体が STABLE / CHAIN / EFFECT /
    OJAMA_FALL / TSUMO_FALL / MENU / SKIP のいずれかを示す.
    """

    def __init__(
        self,
        frame_idx: int,
        time_sec: float,
        patches_1p: dict[tuple[int, int], np.ndarray],
        patches_2p: dict[tuple[int, int], np.ndarray],
        predictions_1p: dict[tuple[int, int], int],
        predictions_2p: dict[tuple[int, int], int],
        display_patches_1p: dict[tuple[int, int], np.ndarray] | None = None,
        display_patches_2p: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> None:
        self.frame_idx = frame_idx
        self.time_sec = time_sec
        # 学習用 patch (= cell_sample_rect、 下寄せ、 CNN 互換)
        self.patches_1p = patches_1p
        self.patches_2p = patches_2p
        # 表示用 patch (= cell 全体、 上下も見える). None なら学習用と同じを使う.
        self.display_patches_1p = (
            display_patches_1p if display_patches_1p is not None else patches_1p
        )
        self.display_patches_2p = (
            display_patches_2p if display_patches_2p is not None else patches_2p
        )
        self.predictions_1p = predictions_1p
        self.predictions_2p = predictions_2p
        # 確定ラベル (= 全 cell 必須). 初期値 = CNN 予測.
        self.labels_1p: dict[tuple[int, int], int] = dict(predictions_1p)
        self.labels_2p: dict[tuple[int, int], int] = dict(predictions_2p)
        # 修正済 cell の set (= 視覚的に色枠で強調)
        self.modified_1p: set[tuple[int, int]] = set()
        self.modified_2p: set[tuple[int, int]] = set()
        # 動作状態ラベル (X-2、 1P/2P 独立、 排他).
        # 例: 1P=CHAIN かつ 2P=TSUMO_FALL のような両立シナリオに対応.
        # SKIP の場合はその side の cell ラベルは保存しない.
        self.frame_state_1p: str = FRAME_STATE_STABLE
        self.frame_state_2p: str = FRAME_STATE_STABLE
        # 演出フラグ (= 動作状態と独立、 ON/OFF).
        # 「CHAIN + 演出あり」 「TSUMO_FALL + 演出あり」 等の両立に対応.
        self.effect_1p: bool = False
        self.effect_2p: bool = False

    # 旧 API 互換: 単一 frame_state プロパティは 1P/2P 同期で扱う
    @property
    def frame_state(self) -> str:
        return self.frame_state_1p

    @frame_state.setter
    def frame_state(self, value: str) -> None:
        self.frame_state_1p = value
        self.frame_state_2p = value

    def set_label(
        self, side: str, row: int, col: int, color: int,
    ) -> None:
        if side == "1P":
            self.labels_1p[(row, col)] = color
            self.modified_1p.add((row, col))
        else:
            self.labels_2p[(row, col)] = color
            self.modified_2p.add((row, col))

    def get_label(self, side: str, row: int, col: int) -> int:
        labels = self.labels_1p if side == "1P" else self.labels_2p
        return labels.get((row, col), COLOR_UNKNOWN)

    def get_patch(
        self, side: str, row: int, col: int,
    ) -> np.ndarray | None:
        """学習用 patch (= cell_sample_rect 領域).

        ラベリング GUI で表示用が欲しい場合は get_display_patch を使う.
        """
        patches = self.patches_1p if side == "1P" else self.patches_2p
        return patches.get((row, col))

    def get_display_patch(
        self, side: str, row: int, col: int,
    ) -> np.ndarray | None:
        """ラベリング GUI 表示用 patch (= cell 全体).

        上部 row も cell 全体が見える. 視認性重視.
        """
        patches = (
            self.display_patches_1p if side == "1P"
            else self.display_patches_2p
        )
        return patches.get((row, col))


# ============================
# Cell パッチ抽出 + CNN 予測
# ============================


def extract_cell_patches(
    frame_1080p: np.ndarray, region: Any,
) -> dict[tuple[int, int], np.ndarray]:
    """1 frame から指定盤面の各 cell の BGR patch を抽出 (= 学習データ用).

    BoardRegion.cell_sample_rect (= 上部 row は下寄せ) で切り出し.
    CNN 入力サイズに合わせた領域、 既存 ColorClassifier と互換.

    Returns:
        {(row, col): patch} (row = 1..12, col = 0..5).
    """
    patches: dict[tuple[int, int], np.ndarray] = {}
    for row in range(VISIBLE_ROW_START, VISIBLE_ROW_END):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_1080p.shape[1], x2)
            y2 = min(frame_1080p.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            patches[(row, col)] = frame_1080p[y1:y2, x1:x2].copy()
    return patches


def extract_cell_patches_for_display(
    frame_1080p: np.ndarray, region: Any,
) -> dict[tuple[int, int], np.ndarray]:
    """1 frame から各 cell の **表示用** BGR patch を抽出.

    学習用 (extract_cell_patches) は CNN 互換で下寄せだが、 ラベリング GUI では
    cell 全体 (= 中心 + 全幅高さ) を表示する方が視認性が高い. 上部 row のぷよの
    上半分も見える.
    """
    patches: dict[tuple[int, int], np.ndarray] = {}
    half_w = int(region.cell_width // 2)
    half_h = int(region.cell_height // 2)
    for row in range(VISIBLE_ROW_START, VISIBLE_ROW_END):
        for col in range(BOARD_COLS):
            cx, cy = region.cell_center(row, col)
            x1 = max(0, int(cx) - half_w)
            y1 = max(0, int(cy) - half_h)
            x2 = min(frame_1080p.shape[1], int(cx) + half_w)
            y2 = min(frame_1080p.shape[0], int(cy) + half_h)
            if x2 <= x1 or y2 <= y1:
                continue
            patches[(row, col)] = frame_1080p[y1:y2, x1:x2].copy()
    return patches


def predict_cells(
    patches: dict[tuple[int, int], np.ndarray],
    classifier: ColorClassifier,
) -> dict[tuple[int, int], int]:
    """ColorClassifier で各 cell の色を予測."""
    preds: dict[tuple[int, int], int] = {}
    for key, patch in patches.items():
        try:
            preds[key] = classifier.classify(patch)
        except Exception:
            preds[key] = COLOR_UNKNOWN
    return preds


# ============================
# UI 描画
# ============================


def render_canvas(
    fl: FrameLabels,
    selected: tuple[str, int, int] | None,
    total_frames: int,
    cur_idx: int,
    save_count: int,
) -> np.ndarray:
    """1 frame の UI canvas を生成."""
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 40)  # 背景色

    # 1P (左)
    _draw_grid(
        canvas, fl, "1P", offset_x=0, selected=selected,
    )
    # 2P (中央)
    _draw_grid(
        canvas, fl, "2P", offset_x=GRID_W + SEP_W, selected=selected,
    )
    # 右ペイン: 拡大プレビュー + 色パレット
    right_x = GRID_W * 2 + SEP_W
    _draw_right_pane(canvas, fl, selected, right_x)

    # ステータス bar
    y0 = GRID_H + 25
    cv2.putText(
        canvas,
        f"Frame {cur_idx + 1}/{total_frames} (idx={fl.frame_idx} t={fl.time_sec:.2f}s)",
        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2,
    )
    cv2.putText(
        canvas,
        f"Modified: 1P={len(fl.modified_1p)} 2P={len(fl.modified_2p)}  Saved frames={save_count}",
        (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )
    cv2.putText(
        canvas,
        "Colors: R(red) Y(yel) P(pur) B(blu) G(grn) O(ojama) E(empty)  /  1-5,9,0 also work",
        (10, y0 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1,
    )
    cv2.putText(
        canvas,
        "Space/]/. = Next  /  [/, = Prev  /  A = ApproveAll  /  U = Undo  /  S = Save  /  Q = Quit",
        (10, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1,
    )
    if selected:
        side, r, c = selected
        pred_color = (
            fl.predictions_1p.get((r, c), COLOR_UNKNOWN)
            if side == "1P"
            else fl.predictions_2p.get((r, c), COLOR_UNKNOWN)
        )
        lbl_color = fl.get_label(side, r, c)
        cv2.putText(
            canvas,
            f"Selected: {side} row={r} col={c} | CNN={COLOR_TO_NAME[pred_color]} | Label={COLOR_TO_NAME[lbl_color]}",
            (10, y0 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2,
        )
    return canvas


# 色パレットボタン定義: (color, name, key_char)
PALETTE_BUTTONS: list[tuple[int, str, str]] = [
    (COLOR_RED, "RED", "R"),
    (COLOR_YELLOW, "YELLOW", "Y"),
    (COLOR_GREEN, "GREEN", "G"),
    (COLOR_BLUE, "BLUE", "B"),
    (COLOR_PURPLE, "PURPLE", "P"),
    (COLOR_OJAMA, "OJAMA", "O"),
    (COLOR_EMPTY, "EMPTY", "E"),
    # UNKNOWN: 落下中ぷよ・演出で見えない cell 用. 学習から除外される.
    (COLOR_UNKNOWN, "UNKNOWN", "X"),
]


def _draw_right_pane(
    canvas: np.ndarray, fl: FrameLabels,
    selected: tuple[str, int, int] | None, offset_x: int,
) -> None:
    """右ペイン: 拡大プレビュー + 色パレット."""
    # 拡大プレビュー (上部)
    px = offset_x + 20
    py = 20
    cv2.putText(
        canvas, "SELECTED CELL",
        (px, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1,
    )
    py += 28
    if selected is not None:
        side, r, c = selected
        patch = fl.get_patch(side, r, c)
        if patch is not None:
            resized = cv2.resize(patch, (PREVIEW_PX, PREVIEW_PX),
                                 interpolation=cv2.INTER_NEAREST)
            canvas[py:py + PREVIEW_PX, px:px + PREVIEW_PX] = resized
            cv2.rectangle(
                canvas, (px, py), (px + PREVIEW_PX, py + PREVIEW_PX),
                (200, 200, 200), 1,
            )
    else:
        cv2.rectangle(
            canvas, (px, py), (px + PREVIEW_PX, py + PREVIEW_PX),
            (60, 60, 80), -1,
        )
        cv2.putText(
            canvas, "(no selection)",
            (px + 20, py + PREVIEW_PX // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1,
        )
    py_after_prev = py + PREVIEW_PX + 20

    # 色パレット (= 各ボタンは click 可能、 色ヘッダ + 頭文字)
    cv2.putText(
        canvas, "COLOR PALETTE",
        (px, py_after_prev + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1,
    )
    py_after_prev += 28
    for i, (color, name, key) in enumerate(PALETTE_BUTTONS):
        bx1 = px + (i % 2) * (PALETTE_BTN_W + 10)
        by1 = py_after_prev + (i // 2) * (PALETTE_BTN_H + 8)
        bx2 = bx1 + PALETTE_BTN_W
        by2 = by1 + PALETTE_BTN_H
        bgr = COLOR_TO_BGR.get(color, (180, 180, 180))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bgr, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (220, 220, 220), 1)
        cv2.putText(
            canvas, name, (bx1 + 8, by1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
        )
        cv2.putText(
            canvas, f"[{key}]", (bx1 + 8, by1 + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
        )


def get_palette_button_rect(
    btn_idx: int,
) -> tuple[int, int, int, int]:
    """色パレットボタン btn_idx の (x1, y1, x2, y2) を返す (mouse click 判定用)."""
    right_x = GRID_W * 2 + SEP_W
    px = right_x + 20
    # _draw_right_pane と完全一致する座標計算
    py_after_prev = 20 + 28 + PREVIEW_PX + 20 + 28
    bx1 = px + (btn_idx % 2) * (PALETTE_BTN_W + 10)
    by1 = py_after_prev + (btn_idx // 2) * (PALETTE_BTN_H + 8)
    return bx1, by1, bx1 + PALETTE_BTN_W, by1 + PALETTE_BTN_H


def _draw_grid(
    canvas: np.ndarray,
    fl: FrameLabels,
    side: str,
    offset_x: int,
    selected: tuple[str, int, int] | None,
) -> None:
    """指定 side の grid を canvas に描画."""
    for row in range(VISIBLE_ROW_START, VISIBLE_ROW_END):
        for col in range(BOARD_COLS):
            visible_row = row - VISIBLE_ROW_START
            x1 = offset_x + col * CELL_THUMB_W
            y1 = visible_row * CELL_THUMB_H
            x2 = x1 + CELL_THUMB_W
            y2 = y1 + CELL_THUMB_H

            patch = fl.get_patch(side, row, col)
            if patch is not None:
                resized = cv2.resize(patch, (CELL_THUMB_W, CELL_THUMB_H))
                canvas[y1:y2, x1:x2] = resized
            else:
                canvas[y1:y2, x1:x2] = (30, 30, 30)

            # ラベル色を四角で表示 (下端、 cell の 25%)
            lbl_color = fl.get_label(side, row, col)
            bar_h = int(CELL_THUMB_H * 0.25)
            bgr = COLOR_TO_BGR.get(lbl_color, (200, 200, 200))
            cv2.rectangle(
                canvas,
                (x1, y2 - bar_h), (x2, y2),
                bgr, thickness=-1,
            )
            cv2.putText(
                canvas, COLOR_TO_NAME[lbl_color],
                (x1 + 4, y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

            # 修正済セルは黄色枠
            modified = (
                fl.modified_1p if side == "1P" else fl.modified_2p
            )
            if (row, col) in modified:
                cv2.rectangle(
                    canvas, (x1, y1), (x2 - 1, y2 - 1),
                    (0, 255, 255), thickness=2,
                )
            # CNN 予測と label が異なる cell は赤枠 (= 修正候補強調)
            pred = (
                fl.predictions_1p.get((row, col), COLOR_UNKNOWN)
                if side == "1P"
                else fl.predictions_2p.get((row, col), COLOR_UNKNOWN)
            )
            if pred != lbl_color and (row, col) not in modified:
                cv2.rectangle(
                    canvas, (x1 + 1, y1 + 1), (x2 - 2, y2 - 2),
                    (0, 0, 255), thickness=1,
                )

            # 選択中 cell は青枠
            if selected == (side, row, col):
                cv2.rectangle(
                    canvas, (x1 + 2, y1 + 2), (x2 - 3, y2 - 3),
                    (255, 100, 0), thickness=3,
                )

    # side label
    cv2.putText(
        canvas, side, (offset_x + 10, GRID_H - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )


# ============================
# マウス callback
# ============================


class MouseState:
    def __init__(self) -> None:
        self.selected: tuple[str, int, int] | None = None
        # 色パレット click イベントを伝える (= main loop が処理)
        self.pending_palette_color: int | None = None


def make_mouse_callback(state: MouseState) -> Any:
    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        right_pane_x = GRID_W * 2 + SEP_W
        # 色パレットボタン click 判定 (右ペイン内)
        if x >= right_pane_x and y < GRID_H:
            for idx, (color, _name, _key) in enumerate(PALETTE_BUTTONS):
                bx1, by1, bx2, by2 = get_palette_button_rect(idx)
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    state.pending_palette_color = color
                    return
            # 右ペインだが palette 外 = 無視
            return
        # grid 領域: side 判定
        if y >= GRID_H:
            return
        if x < GRID_W:
            side = "1P"
            x_rel = x
        elif GRID_W + SEP_W <= x < GRID_W * 2 + SEP_W:
            side = "2P"
            x_rel = x - GRID_W - SEP_W
        else:
            return
        col = x_rel // CELL_THUMB_W
        visible_row = y // CELL_THUMB_H
        row = visible_row + VISIBLE_ROW_START
        if 0 <= col < BOARD_COLS and VISIBLE_ROW_START <= row < VISIBLE_ROW_END:
            state.selected = (side, int(row), int(col))
    return on_mouse


# ============================
# 保存
# ============================


def _apply_color(
    fl: FrameLabels, side: str, r: int, c: int, color: int,
    undo_stack: list[tuple[int, str, int, int, int, bool]],
) -> None:
    """選択 cell に色付与 (undo stack 更新付き)."""
    labels = fl.labels_1p if side == "1P" else fl.labels_2p
    modified = fl.modified_1p if side == "1P" else fl.modified_2p
    prev_color = labels.get((r, c), COLOR_UNKNOWN)
    prev_mod = (r, c) in modified
    undo_stack.append((fl.frame_idx, side, r, c, prev_color, prev_mod))
    labels[(r, c)] = color
    modified.add((r, c))


def save_frame_labels(
    fl: FrameLabels, store: LabelStore, video_id: str,
) -> int:
    """FrameLabels の 1P + 2P 全 cell + 1P/2P frame state を一括保存.

    X-2 (1P/2P 独立版): frame_state_<side> が SKIP の場合はその side の cell
    ラベルを保存しない. frame_state は 1P/2P 別々に各 1 件 保存.

    Returns:
        保存した PseudoLabelSample の総数.
    """
    samples: list[PseudoLabelSample] = []
    for side, labels, patches, state, effect in (
        ("1P", fl.labels_1p, fl.patches_1p, fl.frame_state_1p, fl.effect_1p),
        ("2P", fl.labels_2p, fl.patches_2p, fl.frame_state_2p, fl.effect_2p),
    ):
        # SKIP frame では cell ラベル保存しない (= その side のみ)
        if state != FRAME_STATE_SKIP:
            for (row, col), color in labels.items():
                patch = patches.get((row, col))
                if patch is None:
                    continue
                samples.append(PseudoLabelSample(
                    component=COMPONENT_CELL,
                    timestamp=fl.time_sec,
                    input_data={"patch": patch},
                    label=int(color),
                    confidence=1.0,
                    metadata={
                        "video_id": video_id,
                        "frame_idx": int(fl.frame_idx),
                        "row": int(row),
                        "col": int(col),
                        "side": side,
                        "manual": True,
                        "frame_state": state,
                        "effect": bool(effect),
                    },
                ))
        # X-2: frame state ラベル (= side ごとに 1 件、 SKIP も含む).
        # 動作状態 + 演出フラグ を組み合わせて 1 件保存.
        samples.append(PseudoLabelSample(
            component=COMPONENT_FRAME_STATE,
            timestamp=fl.time_sec,
            input_data=None,
            label=state,
            confidence=1.0,
            metadata={
                "video_id": video_id,
                "frame_idx": int(fl.frame_idx),
                "side": side,
                "manual": True,
                "effect": bool(effect),
            },
        ))
    if samples:
        store.append(samples)
    return len(samples)


# ============================
# サンプリング
# ============================


def sample_frames(
    video_path: Path, start_sec: float, end_sec: float,
    interval_sec: float,
) -> list[tuple[int, float, np.ndarray]]:
    """動画から指定範囲を interval ごとにサンプリングし、 (frame_idx, time_sec,
    1080p frame) を返す."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames: list[tuple[int, float, np.ndarray]] = []
    cur_sec = float(start_sec)
    while cur_sec <= end_sec:
        target_idx = int(round(cur_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # 1920x1080 へリサイズ (= image_reader と同じ前提)
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        frames.append((target_idx, cur_sec, frame))
        cur_sec += interval_sec
    cap.release()
    return frames


# ============================
# メイン
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", type=str, required=True,
                        help="data/pseudo_labels/<video_id>/cell.jsonl に保存")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=75.0)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--store-root", type=Path,
                        default=Path("data/pseudo_labels"))
    args = parser.parse_args()

    print(f"[sampling] {args.video} sec=[{args.start_sec}, {args.end_sec}] interval={args.interval_sec}")
    sampled = sample_frames(
        args.video, args.start_sec, args.end_sec, args.interval_sec,
    )
    print(f"[sampling] got {len(sampled)} frames")
    if not sampled:
        print("[error] no frames sampled")
        return 1

    classifier = ColorClassifier()
    print("[init] ColorClassifier (HSV-only) loaded")
    p1 = DEFAULT_P1_REGION
    p2 = DEFAULT_P2_REGION

    frame_labels: list[FrameLabels] = []
    print("[init] extracting patches + predicting cells ...")
    for fi, t_sec, frame in sampled:
        patches_1p = extract_cell_patches(frame, p1)
        patches_2p = extract_cell_patches(frame, p2)
        disp_1p = extract_cell_patches_for_display(frame, p1)
        disp_2p = extract_cell_patches_for_display(frame, p2)
        pred_1p = predict_cells(patches_1p, classifier)
        pred_2p = predict_cells(patches_2p, classifier)
        frame_labels.append(FrameLabels(
            frame_idx=fi, time_sec=t_sec,
            patches_1p=patches_1p, patches_2p=patches_2p,
            predictions_1p=pred_1p, predictions_2p=pred_2p,
            display_patches_1p=disp_1p, display_patches_2p=disp_2p,
        ))
    print(f"[init] done. {len(frame_labels)} frames ready")

    store = LabelStore(video_id=args.video_id, root=args.store_root)
    cur_idx = 0
    saved_frames = 0
    mouse_state = MouseState()
    win_name = "label_cells"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, make_mouse_callback(mouse_state))

    # undo stack: (frame_idx, side, row, col, prev_color, prev_modified_flag)
    undo_stack: list[tuple[int, str, int, int, int, bool]] = []

    while True:
        fl = frame_labels[cur_idx]
        canvas = render_canvas(
            fl, mouse_state.selected, len(frame_labels), cur_idx, saved_frames,
        )
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:  # no input
            continue
        if key == ord("q") or key == ord("Q"):
            # 終了時にも未保存 frame は保存
            for i in range(cur_idx, len(frame_labels)):
                pass  # 操作明示の方が良いので auto 保存はしない
            print(f"[done] saved {saved_frames} frames to {args.store_root}/{args.video_id}/cell.jsonl")
            break
        if key == 32 or key == ord("]") or key == ord("."):  # Space / ] / . : 次 frame
            n = save_frame_labels(fl, store, args.video_id)
            saved_frames += 1
            print(f"[save] frame {fl.frame_idx} t={fl.time_sec:.2f}s : {n} cells (cumulative frames={saved_frames})")
            if cur_idx < len(frame_labels) - 1:
                cur_idx += 1
                mouse_state.selected = None
            else:
                print("[info] reached last frame.")
        elif key == ord("[") or key == ord(","):  # [ / , : 前 frame (B=BLUE と衝突回避)
            if cur_idx > 0:
                cur_idx -= 1
                mouse_state.selected = None
        elif key == ord("a") or key == ord("A"):
            # 全 cell を CNN 予測のまま承認 (実質は labels = predictions のまま、
            # ただし「修正済」 マークはつけない). 視覚的には全 cell 確定扱い.
            pass  # no-op: labels は既に予測値で初期化済
        elif key == ord("s") or key == ord("S"):
            n = save_frame_labels(fl, store, args.video_id)
            saved_frames += 1
            print(f"[save] frame {fl.frame_idx} t={fl.time_sec:.2f}s : {n} cells (manual save)")
        elif key == ord("u") or key == ord("U"):
            if undo_stack:
                fi, side, r, c, prev_color, prev_mod = undo_stack.pop()
                labels = fl.labels_1p if side == "1P" else fl.labels_2p
                modified = fl.modified_1p if side == "1P" else fl.modified_2p
                labels[(r, c)] = prev_color
                if not prev_mod:
                    modified.discard((r, c))
        elif key in KEY_TO_COLOR and mouse_state.selected is not None:
            side, r, c = mouse_state.selected
            color = KEY_TO_COLOR[key]
            _apply_color(fl, side, r, c, color, undo_stack)

        # 色パレット click 後の color 適用 (= マウスで色選択)
        if (
            mouse_state.pending_palette_color is not None
            and mouse_state.selected is not None
        ):
            side, r, c = mouse_state.selected
            color = mouse_state.pending_palette_color
            _apply_color(fl, side, r, c, color, undo_stack)
            mouse_state.pending_palette_color = None

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
