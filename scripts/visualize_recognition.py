"""認識+物理推論結果を盤面に重畳する可視化動画を生成する.

各セルに認識色 (赤/青/緑/黄/紫/お/空/?) を文字で描画し、
盤面外枠に state machine の現在状態 (STABLE/CHAIN/TSUMO_FALL/...) を描画する.

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \\
        --video data/evaluation_videos/v28_clip60s.mp4 \\
        --output data/evaluation_videos/v28_recognition_viz.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 描画定数
# ============================
# 盤面 ROI (calibration_video01.json より)
P1_ROI_X = 282
P1_ROI_Y = 160
P2_ROI_X = 1258
P2_ROI_Y = 160
ROI_W = 384
ROI_H = 720
N_VISIBLE_ROWS = 12
CELL_W = ROI_W // BOARD_COLS  # 64 px
CELL_H = ROI_H // N_VISIBLE_ROWS  # 60 px

# 色記号 (コンパクト表示)
COLOR_SYMBOLS = {
    COLOR_EMPTY: "",
    COLOR_RED: "R",
    COLOR_BLUE: "B",
    COLOR_GREEN: "G",
    COLOR_YELLOW: "Y",
    COLOR_PURPLE: "P",
    COLOR_OJAMA: "O",
    COLOR_UNKNOWN: "?",
}
# BGR 色 (cv2)
COLOR_BGR = {
    COLOR_EMPTY: (60, 60, 60),
    COLOR_RED: (50, 50, 240),
    COLOR_BLUE: (240, 100, 50),
    COLOR_GREEN: (50, 220, 50),
    COLOR_YELLOW: (50, 220, 240),
    COLOR_PURPLE: (200, 50, 200),
    COLOR_OJAMA: (180, 180, 180),
    COLOR_UNKNOWN: (255, 255, 255),
}

# State machine state ごとの枠色
STATE_COLOR = {
    BoardState.STABLE: (0, 255, 0),         # green = OK
    BoardState.TSUMO_FALL: (0, 200, 255),   # orange
    BoardState.CHAIN: (200, 100, 255),      # pink/purple
    BoardState.OJAMA_FALL: (255, 200, 0),   # cyan
    BoardState.MENU: (128, 128, 128),       # gray
    BoardState.EFFECT: (255, 0, 255),       # magenta (全消し等)
}

# 描画フォント
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE_CELL = 0.7
FONT_SCALE_STATE = 0.9
FONT_THICKNESS = 2

# サンプリング間隔 (秒)
DEFAULT_SAMPLE_INTERVAL = 0.033  # 30 fps 認識 (= cycle 71p 2026-05-13 ユーザー要望)


def draw_cell_overlay(
    frame: np.ndarray, board: Board, roi_x: int, roi_y: int,
) -> None:
    """盤面 1 つに対し、各 cell の色 symbol を重畳する.

    可視 12 行のみ描画 (隠し段 row 0 は省略)。
    文字色は常に白、黒太縁で puyo 背景と同色化を回避。
    """
    if board is None:
        return
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = int(board.get(row, col))
            symbol = COLOR_SYMBOLS.get(color, "?")
            if not symbol:
                continue  # EMPTY は描画しない
            # cell 中心座標 (visible row index = row - HIDDEN_ROWS)
            display_row = row - HIDDEN_ROWS
            cx = roi_x + col * CELL_W + CELL_W // 2
            cy = roi_y + display_row * CELL_H + CELL_H // 2
            # 文字サイズ調整
            (tw, th), _ = cv2.getTextSize(
                symbol, FONT, FONT_SCALE_CELL, FONT_THICKNESS,
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            # 黒太縁 (puyo 背景と同色化を回避するため)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (0, 0, 0), FONT_THICKNESS + 4, cv2.LINE_AA,
            )
            # 白文字 (常に視認性確保)
            cv2.putText(
                frame, symbol, (tx, ty), FONT,
                FONT_SCALE_CELL, (255, 255, 255),
                FONT_THICKNESS, cv2.LINE_AA,
            )


def draw_state_label(
    frame: np.ndarray, state: BoardState, roi_x: int, roi_y: int,
    score: int = 0, label_prefix: str = "",
) -> None:
    """ROI 上方に state ラベルを描画する."""
    color = STATE_COLOR.get(state, (255, 255, 255))
    text = f"{label_prefix}{state.value}"
    if score > 0:
        text += f" score={score}"
    # 影
    cv2.putText(
        frame, text, (roi_x + 4, roi_y - 12), FONT,
        FONT_SCALE_STATE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (roi_x + 3, roi_y - 13), FONT,
        FONT_SCALE_STATE, color, FONT_THICKNESS, cv2.LINE_AA,
    )
    # ROI 枠
    cv2.rectangle(
        frame, (roi_x, roi_y), (roi_x + ROI_W, roi_y + ROI_H),
        color, 2,
    )


def draw_global_info(
    frame: np.ndarray, frame_idx: int, t_sec: float,
    p1_state: BoardState, p2_state: BoardState,
) -> None:
    """画面上部に時刻 + 状態を描画."""
    text = f"frame={frame_idx} t={t_sec:.2f}s 1P={p1_state.value} 2P={p2_state.value}"
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (20, 30), FONT, 0.8,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


_HSV_DB_ROOT = Path("data/per_video_hsv_ranges")
_HSV_MERGED_DEFAULT = _HSV_DB_ROOT / "_merged_default.json"

# 2範囲以上で定義されている色 → per_video inject 後に DEFAULT の補完範囲を保証する
# (赤は H=0-13 と H=166-180 の循環2範囲。per_video は高側のみ学習しがちなので
#  低側 H=0-13 が失われると赤を系統的に miss する)
_CIRCULAR_GUARD_COLORS: tuple[int, ...] = (COLOR_RED,)


def _ensure_circular_ranges_guard(classifier: object) -> None:
    """per_video inject 後に DEFAULT の循環補完範囲が欠落していないか確認し補完する。

    赤 (COLOR_RED=1) は H=0-13 と H=166-180 の2範囲で循環 Hue をカバーする。
    per_video inject が H=166-180 側のみ学習した場合、append=True でも
    DEFAULT の H=0-13 側が存在する前提。ただし inject 経路のバグや
    将来的な変更で欠落するリスクを guard する。

    Args:
        classifier: ColorClassifier インスタンス (_ranges 属性を持つオブジェクト)。
    """
    from src.image_reader import DEFAULT_COLOR_RANGES, HsvRange
    if not hasattr(classifier, "_ranges"):
        return
    for color in _CIRCULAR_GUARD_COLORS:
        if color not in DEFAULT_COLOR_RANGES:
            continue
        default_rngs = DEFAULT_COLOR_RANGES[color]
        if len(default_rngs) < 2:
            # DEFAULT が1範囲なら循環問題なし
            continue
        current: list[HsvRange] = list(classifier._ranges.get(color, []))
        for dflt in default_rngs:
            already = any(
                r.h_min == dflt.h_min and r.h_max == dflt.h_max
                for r in current
            )
            if not already:
                current.append(dflt)
                print(
                    f"[viz] circular_guard: color={color} "
                    f"H=[{dflt.h_min},{dflt.h_max}] を補完"
                )
        classifier._ranges[color] = current


def resolve_hsv_path(video_path: Path) -> Path:
    """動画ファイル名から動画 ID を抽出し、 per-video HSV JSON を自動選択する。

    優先順位:
      1. video_path のファイル名先頭から "(v[0-9]+)" を抽出
      2. data/per_video_hsv_ranges/{video_id}.json が存在 → それを返す (per-video 直接 inject)
      3. 不在 → _merged_default.json を返す (fallback)

    案 K (2026-05-24): 38 動画 union の背景誤認問題を per-video 直接 inject で回避。
    """
    import re
    m = re.match(r"(v\d+)", video_path.name)
    if m:
        candidate = _HSV_DB_ROOT / f"{m.group(1)}.json"
        if candidate.exists():
            return candidate
    return _HSV_MERGED_DEFAULT


def _extract_raw_hsv_board(
    reader: object,
    frame: "np.ndarray",
    region: object,
) -> list[list[int]]:
    """ImageReader から HSV-only 経路の直出力 board を取得する。

    HybridClassifier の _hsv 属性、または ColorClassifier 直参照で
    全 visible cell に HSV 分類を適用して 13×6 grid を返す。
    bg_fp / CNN 経路をバイパスした純 HSV 結果。

    Args:
        reader: ImageReader インスタンス。
        frame: 1920x1080 BGR フレーム。
        region: BoardRegion インスタンス。

    Returns:
        13×6 の int grid (COLOR_* 定数)。
    """
    import cv2 as _cv2
    import numpy as _np
    from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, HIDDEN_ROWS

    # HSV 分類器の取得 (HybridClassifier._hsv or ColorClassifier 直)
    classifier_raw = getattr(reader, "_classifier", None)
    hsv_cls = getattr(classifier_raw, "_hsv", classifier_raw)

    img_h, img_w = frame.shape[:2]
    grid: list[list[int]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[int] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(COLOR_EMPTY)
                continue
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            patch = frame[y1:y2, x1:x2]
            try:
                color = int(hsv_cls.classify(patch)) if hsv_cls is not None else COLOR_EMPTY
            except Exception:
                color = COLOR_EMPTY
            row_vals.append(color)
        grid.append(row_vals)
    return grid


def _extract_bg_fp_distance_grid(
    reader: object,
    frame: "np.ndarray",
    region: object,
    side: str,
) -> list[list[float]] | None:
    """各 cell の bg_fp 距離 (float) を 13×6 grid で返す。

    bg_fp が未採取の場合は None を返す。
    hidden rows (row < HIDDEN_ROWS) は -1.0 で埋める。

    Args:
        reader: ImageReader インスタンス。
        frame: 1920x1080 BGR フレーム。
        region: BoardRegion インスタンス。
        side: "1P" or "2P"。

    Returns:
        13×6 の float grid、または None (bg_fp 未採取)。
    """
    import cv2 as _cv2
    import numpy as _np
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS

    bg_fp = getattr(reader, "_bg_fp_p1" if side == "1P" else "_bg_fp_p2", None)
    if bg_fp is None:
        return None

    try:
        from src.background_fingerprint import CellFingerprint
    except Exception:
        return None

    hsv_full = _cv2.cvtColor(frame, _cv2.COLOR_BGR2HSV)
    img_h, img_w = frame.shape[:2]

    grid: list[list[float]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[float] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(-1.0)
                continue
            visible_row = row - HIDDEN_ROWS
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            hsv_patch = hsv_full[y1:y2, x1:x2]
            if hsv_patch.size == 0:
                row_vals.append(-1.0)
                continue
            h_med = int(_np.median(hsv_patch[:, :, 0]))
            s_med = int(_np.median(hsv_patch[:, :, 1]))
            v_med = int(_np.median(hsv_patch[:, :, 2]))
            cur_fp = CellFingerprint(h_med, s_med, v_med)
            try:
                bg_cell = bg_fp.cell_at(visible_row, col)
                dist = float(cur_fp.distance_to(bg_cell))
            except Exception:
                dist = -1.0
            row_vals.append(dist)
        grid.append(row_vals)
    return grid


def _extract_tier1_threshold_grid(
    reader: object,
    region: object,
) -> list[list[float]]:
    """各 cell の tier1 閾値 (float) を 13×6 grid で返す。

    _resolve_tier1_threshold() が存在しない場合は全 cell DEFAULT を使う。
    hidden rows は -1.0 で埋める。

    Args:
        reader: ImageReader インスタンス。
        region: 使用しない (visible_row/col のみで閾値は決まる)。

    Returns:
        13×6 の float grid。
    """
    from src.board import BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS
    from src.image_reader import BG_EXTREME_THRESHOLD_DEFAULT

    resolve_fn = getattr(reader, "_resolve_tier1_threshold", None)

    grid: list[list[float]] = []
    for row in range(BOARD_ROWS):
        row_vals: list[float] = []
        for col in range(BOARD_COLS):
            if row < HIDDEN_ROWS:
                row_vals.append(-1.0)
                continue
            visible_row = row - HIDDEN_ROWS
            if resolve_fn is not None:
                try:
                    thresh = float(resolve_fn(visible_row, col))
                except Exception:
                    thresh = BG_EXTREME_THRESHOLD_DEFAULT
            else:
                thresh = BG_EXTREME_THRESHOLD_DEFAULT
            row_vals.append(thresh)
        grid.append(row_vals)
    return grid


def _board_diff_cells(
    board_before: "Board | None",
    board_after: "Board | None",
) -> list[list[int]]:
    """2 つの board の差分 cell を [[row, col, before, after], ...] で返す。

    None の場合は空リストを返す。

    Args:
        board_before: 変更前の Board。
        board_after: 変更後の Board。

    Returns:
        差分 cell リスト。各要素 = [row, col, color_before, color_after]。
    """
    from src.board import BOARD_COLS, BOARD_ROWS
    if board_before is None or board_after is None:
        return []
    diffs: list[list[int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv = int(board_before.get(r, c))
            av = int(board_after.get(r, c))
            if bv != av:
                diffs.append([r, c, bv, av])
    return diffs


def _build_detailed_log_entry(
    fi: int,
    t_sec: float,
    result: object,
    frame: "np.ndarray",
    pipeline: object,
    prev_p1_confirmed: "Board | None",
    prev_p2_confirmed: "Board | None",
) -> dict:
    """詳細 board log の 1 エントリを生成する。

    SideResult.cnn_board (= ImageReader 直出力)、raw_hsv_board、
    bg_fp_distance_grid、tier1_threshold_grid、constraint_fill_changed_cells
    (= cnn_board → confirmed_board の差分) を記録する。

    Args:
        fi: frame index。
        t_sec: 経過秒数。
        result: PipelineResult インスタンス。
        frame: 1920x1080 BGR フレーム。
        pipeline: RecognitionPipeline インスタンス。
        prev_p1_confirmed: 前フレームの 1P confirmed_board (差分計算用)。
        prev_p2_confirmed: 前フレームの 2P confirmed_board (差分計算用)。

    Returns:
        JSON シリアライズ可能な dict。
    """
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

    reader = getattr(pipeline, "_reader", None)

    # raw_cnn_board = SideResult.cnn_board (= ImageReader 直出力、pipeline 内で保持済)
    p1_cnn = getattr(result.p1, "cnn_board", None)
    p2_cnn = getattr(result.p2, "cnn_board", None)

    # raw_hsv_board: HSV-only 経路の全 cell 再分類
    p1_hsv_grid: list[list[int]] | None = None
    p2_hsv_grid: list[list[int]] | None = None
    if reader is not None:
        try:
            p1_hsv_grid = _extract_raw_hsv_board(reader, frame, DEFAULT_P1_REGION)
            p2_hsv_grid = _extract_raw_hsv_board(reader, frame, DEFAULT_P2_REGION)
        except Exception:
            pass

    # bg_fp_distance_grid
    p1_bg_dist: list[list[float]] | None = None
    p2_bg_dist: list[list[float]] | None = None
    if reader is not None:
        try:
            p1_bg_dist = _extract_bg_fp_distance_grid(reader, frame, DEFAULT_P1_REGION, "1P")
            p2_bg_dist = _extract_bg_fp_distance_grid(reader, frame, DEFAULT_P2_REGION, "2P")
        except Exception:
            pass

    # tier1_threshold_grid
    p1_tier1: list[list[float]] | None = None
    p2_tier1: list[list[float]] | None = None
    if reader is not None:
        try:
            p1_tier1 = _extract_tier1_threshold_grid(reader, DEFAULT_P1_REGION)
            p2_tier1 = _extract_tier1_threshold_grid(reader, DEFAULT_P2_REGION)
        except Exception:
            pass

    # pre_capture_mode
    pre_capture = bool(getattr(reader, "_pre_capture_mode", False)) if reader else False

    # constraint_fill_changed_cells: cnn_board → confirmed_board の差分
    # (constraint-fill / physics-fix 両方含む「何かが変えた」差分)
    p1_constraint_diff = _board_diff_cells(p1_cnn, result.p1.confirmed_board)
    p2_constraint_diff = _board_diff_cells(p2_cnn, result.p2.confirmed_board)

    # physics_fix_changed_cells: prev_confirmed → confirmed の差分
    p1_physics_diff = _board_diff_cells(prev_p1_confirmed, result.p1.confirmed_board)
    p2_physics_diff = _board_diff_cells(prev_p2_confirmed, result.p2.confirmed_board)

    return {
        "frame_idx": fi,
        "t_sec": t_sec,
        "p1_state": result.p1.state.value,
        "p2_state": result.p2.state.value,
        "p1_confirmed": (
            result.p1.confirmed_board.to_dict()["grid"]
            if result.p1.confirmed_board is not None else None
        ),
        "p2_confirmed": (
            result.p2.confirmed_board.to_dict()["grid"]
            if result.p2.confirmed_board is not None else None
        ),
        # 詳細フィールド
        "p1_raw_cnn_board": p1_cnn.to_dict()["grid"] if p1_cnn is not None else None,
        "p2_raw_cnn_board": p2_cnn.to_dict()["grid"] if p2_cnn is not None else None,
        "p1_raw_hsv_board": p1_hsv_grid,
        "p2_raw_hsv_board": p2_hsv_grid,
        "p1_bg_fp_distance_grid": p1_bg_dist,
        "p2_bg_fp_distance_grid": p2_bg_dist,
        "p1_tier1_threshold_grid": p1_tier1,
        "p2_tier1_threshold_grid": p2_tier1,
        "pre_capture_mode": pre_capture,
        # constraint_fill で変更された cell: [row, col, cnn_color, confirmed_color]
        "p1_constraint_fill_changed_cells": p1_constraint_diff,
        "p2_constraint_fill_changed_cells": p2_constraint_diff,
        # physics_fix で変更された cell: [row, col, prev_confirmed_color, new_confirmed_color]
        "p1_physics_fix_changed_cells": p1_physics_diff,
        "p2_physics_fix_changed_cells": p2_physics_diff,
        # 着地色診断 (2026-06-01 infer_placement 調査用)。
        # TSUMO_FALL→STABLE 着地フレームのみ非 null。
        # falling_pair_old: prev_next_queue[-2] 由来 (従来ロジック)
        # falling_pair_new: _landing_pending[1] 由来 (修正ロジック)
        # source: "landing_pending" | "next_queue_2" | "next_queue_1" | "none"
        "p1_landing_diag": getattr(result.p1, "landing_diag", None),
        "p2_landing_diag": getattr(result.p2, "landing_diag", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-interval", type=float,
        default=DEFAULT_SAMPLE_INTERVAL,
        help="認識処理する frame 間隔 (秒)。出力は元動画の fps を維持し、未サンプル frame は最後の認識結果を保持。",
    )
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="入力動画の処理最大秒数 (0=全部)",
    )
    parser.add_argument("--cnn-model", type=str, default=None)
    parser.add_argument(
        "--hsv-state", type=Path,
        default=None,
        help="動画別 HSV ranges JSON (per_video_hsv_ranges DB)。 起動時に "
             "ColorClassifier.set_color_ranges_from_simple で注入。 "
             "省略時は resolve_hsv_path() が動画 ID から自動選択 "
             "(= 案 K: per-video JSON 優先、 不在時 _merged_default.json fallback)。 "
             "ファイル不在なら silent skip。 無効化したい場合は存在しないパスを渡す。",
    )
    parser.add_argument(
        "--vote-mode", action="store_true",
        help="ColorClassifier を per-pixel 投票方式に切替 (cycle 71)",
    )
    parser.add_argument(
        "--cnn-override-prob", type=float, default=None,
        help="HybridClassifier CNN 採用閾値 (None で default 0.70).",
    )
    parser.add_argument(
        "--mask-ojama-logit", action="store_true",
        help="cycle 32e: CNN の ojama logit を推論時 mask (= argmax 候補から除外)。 "
             "cycle 32e/32f model 等 ojama を学習対象外にした model で使用。",
    )
    parser.add_argument(
        "--no-online-hsv", action="store_true",
        help="OnlineHsvCalibrator を完全無効化 (= 動画中 HSV 学習なし)。 "
             "Step 0 OnlineHsv 効果定量評価用 (2026-05-24)。",
    )
    parser.add_argument(
        "--no-per-video-hsv",
        action="store_true",
        default=False,
        dest="disable_per_video_hsv",
        help="per-video 手調整 HSV inject をスキップする (汎用精度目視確認用)。 "
             "OnlineHsvCalibrator は引き続き動作するため自動 HSV 学習は生きる。 "
             "デフォルト False = 従来挙動完全一致 (backwards compat)。",
    )
    parser.add_argument(
        "--use-puyo-gate", action="store_true",
        help="cycle 32e: PuyoPresenceGate を HybridClassifier 前段に挟む。 "
             "gate=False の patch は HSV-only 経路に倒す (= 背景誤認対策)。",
    )
    parser.add_argument(
        "--use-circle-mask", action="store_true",
        help="cycle 32g: 推論時 patch に円形マスク適用 (学習時と必ず揃える)",
    )
    parser.add_argument(
        "--dump-board-log", type=Path, default=None,
        help="cycle 33 (2026-05-19): 各 frame の confirmed_board / state / "
             "chain_event を JSONL 形式で保存 (= 強化アナリスト用)。 "
             "evaluator が後処理で読み込んで自動評価する。",
    )
    parser.add_argument(
        "--dump-board-log-detailed", type=Path, default=None,
        help="Q1 全力施策: 各 frame の raw_cnn_board / raw_hsv_board / "
             "bg_fp_distance_grid / tier1_threshold_grid / pre_capture_mode / "
             "constraint_fill_changed_cells / physics_fix_changed_cells を含む "
             "詳細 JSONL を保存。 --dump-board-log の superset。 "
             "認識不能 frame の真因解明用 (= extract_unrecognized_frames.py の入力)。",
    )
    # B1 (warmup guard): STABLE 遷移直後 N frame confirmed 凍結 (= A/B 仮説 B1/B3 対応)
    # backwards compat: store_true で default=False、既存挙動は変わらない。
    parser.add_argument(
        "--enable-warmup-guard", action="store_true",
        help="B1 (M1 warmup guard): STABLE 遷移直後 N frame の confirmed board 更新を凍結。 "
             "遷移直後の CNN ノイズ (エフェクト残光・背景誤認) を吸収する。 "
             "A/B 仮説 B1/B3 用 (2026-05-27)。",
    )
    # B2 (bg_fp_force_max_puyo): 背景 FP 採取の緩和条件を絞る (= A/B 仮説 B2/B3 対応)
    # backwards compat: default=None で既存 class attribute 値 (=144) を維持。
    parser.add_argument(
        "--bg-fp-force-max-puyo", type=int, default=None,
        help="B2 (BG_FP_FORCE_MAX_PUYO): 背景 FP 採取緩和の puyo 上限数を上書き。 "
             "None (省略時) = class attribute 値 144 を維持。 "
             "4 に絞ると試合序盤の FP 採取機会が減り誤採取を抑制する。 "
             "A/B 仮説 B2/B3 用 (2026-05-27)。",
    )
    parser.add_argument(
        "--patch-ncc-threshold", type=float, default=None,
        help="PatchBackgroundFingerprint NCC 閾値を上書き (= default は "
             "background_fingerprint.py の PATCH_NCC_EMPTY_THRESHOLD = 0.92)。 "
             "NCC sweep 用 (候補: 0.85, 0.88, 0.90, 0.92)。",
    )
    parser.add_argument(
        "--enable-piece-persistence",
        action="store_true",
        default=False,
        help="B1 PiecePersistenceGuard を有効化 (= STABLE 中 cell 色保護、 散発色ブレ削減)。",
    )
    parser.add_argument(
        "--enable-tier1-warmup",
        action="store_true",
        default=False,
        help="tier1 warmup guard を有効化 (= NON-STABLE → STABLE 遷移直後 "
             f"TIER1_WARMUP_FRAMES={3} frame 間 tier1 を skip し、"
             "ツモ着地直後の cell を tier1 が誤 EMPTY 化するのを防ぐ)。",
    )
    parser.add_argument(
        "--ojama-tier1-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_tier1_warmup",
        help="経路 A': OJAMA_FALL → STABLE 遷移専用の tier1 warmup を制御する。"
             f" OJAMA_TIER1_WARMUP_FRAMES={8} frame 間 tier1 を skip し、"
             "お邪魔消滅後のセル背景化による誤 EMPTY 化 → 列崩壊を防ぐ (v70 対策)。"
             " ライブラリ default=True (有効)。 --no-ojama-tier1-warmup で無効化。",
    )
    parser.add_argument(
        "--constraint-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_constraint_fill",
        help="案2: NEXT 累積制約による色 count 補正 (constraint_fill) を制御する。 "
             "--constraint-fill で有効化、 --no-constraint-fill で無効化。 "
             "ライブラリ default=False (無効)。 "
             "--constraint-fill で有効化して比較 (constraint_fill の net 効果測定用)。",
    )
    parser.add_argument(
        "--t2-highconf-yield",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_t2_highconf_yield",
        help="T2 高確信 yield を制御する。 "
             "STABLE → STABLE 遷移時の prev_stable 上書き (T2) において、 "
             "CNN が現在の confirmed 色を支持しているセルはスキップする。 "
             "infer_placement 誤推論 + T2 自己強化フリーズによる色破壊修正。 "
             "ライブラリ default=True (有効)。 --no-t2-highconf-yield で無効化。",
    )
    parser.add_argument(
        "--infer-empty-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_infer_empty_guard",
        help="infer_placement 空セル hallucination ガードを制御する。 "
             "pattern の非 diff セルが cnn_after で COLOR_EMPTY な候補をスキップし、 "
             "CNN が確信して空なセルへの NEXT 色書込 (hallucination) を防ぐ。 "
             "ライブラリ default=True (有効)。 --no-infer-empty-guard で無効化。",
    )
    parser.add_argument(
        "--game-event-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_game_event_chain_exit",
        help="game-event ベース連鎖終了を制御する。 "
             "CHAIN 状態を timing hold だけでなく「次ツモ変化」または"
             "「連鎖側お邪魔降下」を検知するまで維持する。 "
             "安全弁として CHAIN_MAX_HOLD_SEC (5.0s) 超過で強制終了。 "
             "ライブラリ default=True (有効)。 --no-game-event-chain-exit で無効化。",
    )
    parser.add_argument(
        "--landing-color-fix",
        action="store_true",
        default=False,
        dest="enable_landing_color_fix",
        help="着地色修正 案1: TSUMO_FALL→STABLE 着地時の falling_pair を "
             "prev_next_queue[-2] から _landing_pending (消費済みツモ色) に切り替える。 "
             "slide_motion(R-7) 経由で 1 つ前のツモ色を指してしまう誤色問題の修正。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。 "
             "フラグ OFF でも --dump-board-log-detailed に landing_diag フィールドが記録される。",
    )
    parser.add_argument(
        "--chain-min-display",
        action="store_true",
        default=False,
        dest="enable_chain_min_display",
        help="X1/X4 短連鎖ちらつき対策を有効化。 "
             f"CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC={RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s) + "
             f"短連鎖 game-event exit 抑止 (chain_count < {RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT})。 "
             "enable_game_event_chain_exit と独立フラグ (効果分解のため)。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    parser.add_argument(
        "--hsv-classify-fallback",
        action="store_true",
        default=False,
        dest="enable_hsv_classify_fallback",
        help="HSV 分類 fallback を有効化。 "
             "_classify_next_pair_by_hsv の 2 択強制確定を回避し、 "
             "両候補が拮抗・両候補とも遠い・低彩度 patch の場合は next_pair 素返しにする。 "
             "黄(H26)→赤(H7) 誤分類 (~900 件、 H 差 19) 発火点対策。 "
             "デフォルト OFF = 従来挙動不変 (2 択強制確定、 backwards compat)。",
    )
    parser.add_argument(
        "--landing-observed-color",
        action="store_true",
        default=False,
        dest="enable_landing_observed_color",
        help="真因 A 対処: 着地セルの CNN==HSV 一致色補正を有効化。 "
             "TSUMO_FALL→STABLE 着地時に 2 つの独立認識器 (CNN/HSV) が "
             "一致した着地色を優先し、 falling_pair タイミングずれによる誤色を断つ。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    parser.add_argument(
        "--red-hue-wrap-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_red_hue_wrap_fix",
        help="赤色相折り返し補正を制御する。 "
             "赤ぷよの H 画素が 0-4 と 166-179 に 2 峰分布するため単純 median が "
             "赤/黄境界 (H=13/14) に乗り毎フレームちらつく問題を修正する。 "
             "ライブラリ default=True (有効)。 --no-red-hue-wrap-fix で無効化。",
    )
    parser.add_argument(
        "--specular-robust-saturation",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_specular_robust_saturation",
        help="案D: 光沢ハイライト除外彩度計算を制御する。 "
             "ぷよ表面の白ハイライト画素 (V>=210 かつ S<=60) を彩度 median 計算から除外し、 "
             "光沢球混入による EMPTY 誤判定を防ぐ。 "
             "ライブラリ default=True (有効)。 --no-specular-robust-saturation で無効化。",
    )
    parser.add_argument(
        "--stable-recovery-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_stable_recovery_gate",
        help="設計C 事後復旧ゲートを制御する。 "
             "STABLE 中に confirmed==EMPTY なのに CNN==HSV が同一有効色で "
             "8 フレーム継続したセルを confirmed に復旧する。 "
             "ライブラリ default=True (有効)。 --no-stable-recovery-gate で無効化。",
    )
    # フェーズ A4 (2026-06-02): お邪魔ぷよ視覚的検出・連鎖終了・推論ガード・着地検出
    parser.add_argument(
        "--enable-ojama-visual-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_detection",
        help="フェーズ A4: お邪魔ぷよ視覚的検出を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-visual-detection で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-visual-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_chain_exit",
        help="フェーズ A4: お邪魔ぷよ視覚的検出による連鎖終了判定を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-visual-chain-exit で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-infer-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_infer_guard",
        help="フェーズ A4: お邪魔ぷよ推論ガードを制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-infer-guard で無効化。",
    )
    parser.add_argument(
        "--enable-ojama-settle-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_settle_detection",
        help="フェーズ A4: お邪魔ぷよ着地検出を制御する。 "
             "ライブラリ default=True (有効)。 --no-enable-ojama-settle-detection で無効化。",
    )
    parser.add_argument(
        "--chain-score-early-fire",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_score_early_fire",
        help="機能B: score 急増 CHAIN 早期発火を制御する。 "
             f"True にすると自 side の score_delta >= CHAIN_SCORE_EARLY_FIRE_DELTA={80} "
             "の frame で VideoChainTracker の puyo 減少検知を待たずに即 CHAIN state に突入する。 "
             "OCR 失敗 / score 取得不可時は従来の VideoChainTracker 経路を維持 (OR 追加)。 "
             "ライブラリ default=False (無効)。 --chain-score-early-fire で有効化。",
    )
    parser.add_argument(
        "--chain-exit-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_exit_warmup",
        help="機能C: CHAIN → STABLE 遷移直後の confirmed 凍結を制御する。 "
             f"True にすると CHAIN→STABLE 復帰から CHAIN_EXIT_WARMUP_SEC={0.1}s 間 confirmed "
             "更新を凍結しエフェクト残光色の混入を防ぐ。 "
             "時間ベース実装のため fps 非依存。 "
             "ライブラリ default=False (無効)。 --chain-exit-warmup で有効化。",
    )
    parser.add_argument(
        "--chain-formula-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_chain_formula_detection",
        help="機能D: 連鎖開始 掛け算式 検知を制御する。 "
             "True にすると score ROI の OCR が None (掛け算式表示で NCC conf 低下) かつ "
             "ink_ratio > CHAIN_FORMULA_INK_RATIO_MIN かつ last_score > 0 が "
             "CHAIN_FORMULA_CONSEC_FRAMES 連続で成立した frame で即 CHAIN state に突入する。 "
             "機能B (score 急増経路) と独立フラグ。 "
             "ライブラリ default=True (有効、 2026-06-03 採用)。 --no-chain-formula-detection で無効化。",
    )
    parser.add_argument(
        "--hsv-deferred-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_hsv_deferred_consensus",
        help="案 Y-4: HSV-first commit + deferred consensus を制御する。 "
             "True にすると infer_placement が HSV 拮抗と判定した着地 2 候補を保留し、 "
             "後続フレームの CNN==HSV consensus 投票で確定させる (corruption 65% 起源対策)。 "
             "ライブラリ default=False (無効)。 --hsv-deferred-consensus で有効化。",
    )
    # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    parser.add_argument(
        "--ojama-warning-glow-guard",
        action="store_true",
        default=False,
        dest="enable_ojama_warning_glow_guard",
        help="不具合B 対処: 予告おじゃま発光ガードを有効化する。 "
             "相手連鎖の予告おじゃま演出による盤面上部多色発光を V_high_ratio で検知し、 "
             "STABLE 中の confirmed_board を frozen_board で保護する。 "
             "黄ぷよに発光が重なる黄(4)→おじゃま(9)誤認を防ぐ。 "
             "ライブラリ default=False (無効)。 --ojama-warning-glow-guard で有効化。",
    )
    parser.add_argument(
        "--chain-max-hold-override",
        action="store_true",
        default=False,
        dest="enable_chain_max_hold_override",
        help="案P3: CHAIN_MAX_HOLD_SEC 超過後の ojama 保留を無効化する。 "
             f"active_chain が CHAIN_MAX_HOLD_SEC={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s "
             "超過で強制クリアされた frame では ojama_top_positive による STABLE 復帰保留を "
             "スキップして強制 STABLE に遷移させる (安全弁を本来機能させる)。 "
             "enable_ojama_visual_chain_exit=True と組み合わせて使用する。 "
             "ライブラリ default=False (無効)。 --chain-max-hold-override で有効化。",
    )
    # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (2026-06-05)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    parser.add_argument(
        "--chain-exit-next-signal",
        action="store_true",
        default=False,
        dest="enable_chain_exit_next_signal",
        help="案X*: NextSlide signal による CHAIN 即終了を有効化する。 "
             "(A) 機能D 再点火抑制: 既に CHAIN 中なら 機能D (掛け算式) の発火をスキップし "
             "max_until 延長を止める。 "
             "(B) NextSlide signal (次ツモスライド) 検知で CHAIN を即終了させる。 "
             "warmup 連動: CHAIN_EXIT_WARMUP_SEC 秒間 confirmed 凍結を自動適用。 "
             "真因: ojama_top_positive 保留 + 機能D 再点火による 6.87 秒過剰保持 (v89 1P) を解消。 "
             "ライブラリ default=False (無効)。 --chain-exit-next-signal で有効化。",
    )
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化
    # 2026-06-06 採用: default=True。--no-gravity-settle-state で無効化可。
    parser.add_argument(
        "--gravity-settle-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_gravity_settle_state",
        help="GRAVITY_SETTLE 状態を有効化する (feat/gravity-settle-2026-06-05)。 "
             "連鎖終了直後の重力 settle/着地中を採点外・confirmed 凍結として扱う。 "
             "CHAIN → GRAVITY_SETTLE → STABLE の遷移経路を有効化する。 "
             "案X (--chain-exit-next-signal) との組み合わせを推奨 (内部で自動 ON)。 "
             "default=True (有効、2026-06-06 採用)。 --no-gravity-settle-state で無効化。",
    )
    parser.add_argument(
        "--slide-override-ojama-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_slide_override_ojama_hold",
        help="案γ: CHAIN 中 slide_motion=True (次ツモスライド) が来た場合に "
             "ojama_top_positive による CHAIN 過剰保持 (ojama-hold ガード) を上書きして終了。 "
             "v89 t35.2-39.67 の連鎖過剰保持修正。 "
             "default=True (有効、2026-06-06 採用)。 --no-slide-override-ojama-hold で無効化。",
    )
    args = parser.parse_args()
    # 案 K (2026-05-24): --hsv-state 省略時は動画 ID から自動選択
    if args.hsv_state is None:
        args.hsv_state = resolve_hsv_path(args.video)
        print(f"[viz] HSV auto-resolve: {args.hsv_state} (from {args.video.name})")
    # --no-per-video-hsv: per-video 手調整 HSV inject をスキップする (汎用精度目視確認用)
    # None にすることで下流の inject ブロックを無効化する
    if getattr(args, "disable_per_video_hsv", False):
        args.hsv_state = None
        print(
            "[viz] disable_per_video_hsv=ON "
            "(手調整 per-video HSV inject スキップ: 自動 HSV + merged レンジのみで動作)"
        )
    # cycle 32g: 円形マスクを推論前に有効化
    if args.use_circle_mask:
        from src.patch_classifier import set_circle_mask_enabled
        set_circle_mask_enabled(True)
        print("[viz] use_circle_mask=ON (cycle 32g)")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_sec > 0:
        n_frames = min(n_frames, int(args.max_sec * fps))
    # 認識処理は 1920x1080 前提、出力もそのサイズで揃える
    out_w, out_h = 1920, 1080
    print(f"[input] {args.video} {width}x{height} fps={fps:.1f} frames={n_frames}")
    print(f"[output] {out_w}x{out_h} (resize から書き出し)")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), fourcc, fps, (out_w, out_h),
    )

    # Pipeline (force_in_match=True で MENU 判定スキップ)
    pipeline = RecognitionPipeline.load_default(
        # 2 だと一時的 empty 観測で confirmed_board が誤確定する問題あり
        # (= 26s 1P 盤面 fully filled なのに empty 確定)。 6 で慎重に。
        # サイクル70: 6→3 で遅延 50% 短縮 (= 60fps で 50ms)
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=args.cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        vote_mode=args.vote_mode,
        cnn_override_prob=args.cnn_override_prob,
        mask_ojama_logit=args.mask_ojama_logit,
        use_puyo_gate=args.use_puyo_gate,
        # B1/B2/B3 仮説対応 (2026-05-27): --enable-warmup-guard / --bg-fp-force-max-puyo
        enable_warmup_guard=args.enable_warmup_guard,
        bg_fp_force_max_puyo=args.bg_fp_force_max_puyo,
        # NCC sweep 用 (2026-05-28): --patch-ncc-threshold で閾値上書き
        patch_ncc_threshold=args.patch_ncc_threshold,
        # B1 PiecePersistenceGuard (2026-05-28): --enable-piece-persistence
        enable_piece_persistence=args.enable_piece_persistence,
        # tier1 warmup guard (2026-05-28): --enable-tier1-warmup
        enable_tier1_warmup=args.enable_tier1_warmup,
        # 経路 A' (2026-05-30): --ojama-tier1-warmup で OJAMA 専用 warmup 有効化
        enable_ojama_tier1_warmup=args.enable_ojama_tier1_warmup,
        # 案2 (2026-05-30 / BooleanOptionalAction 整合 2026-06-02):
        # --constraint-fill / --no-constraint-fill で直接制御、反転ロジック除去済み
        enable_constraint_fill=args.enable_constraint_fill,
        # T2 高確信 yield (2026-05-31): --t2-highconf-yield で T2 フリーズ修正を有効化
        enable_t2_highconf_yield=args.enable_t2_highconf_yield,
        # 空セル hallucination ガード (2026-06-01): --infer-empty-guard で有効化
        enable_infer_empty_guard=args.enable_infer_empty_guard,
        # game-event ベース連鎖終了 (2026-06-01): --game-event-chain-exit で有効化
        enable_game_event_chain_exit=args.enable_game_event_chain_exit,
        # 着地色修正 案1 (2026-06-01): --landing-color-fix で有効化
        enable_landing_color_fix=args.enable_landing_color_fix,
        # X1/X4 短連鎖ちらつき対策 (2026-06-01): --chain-min-display で有効化
        enable_chain_min_display=args.enable_chain_min_display,
        # HSV 分類 fallback (fix/v70-zeropatch-redyellow, 2026-06-01): --hsv-classify-fallback で有効化
        enable_hsv_classify_fallback=args.enable_hsv_classify_fallback,
        # 真因 A 対処 (2026-06-01): --landing-observed-color で有効化
        enable_landing_observed_color=args.enable_landing_observed_color,
        # 赤色相折り返し補正 (fix/v70-zeropatch-redyellow, 2026-06-02): --red-hue-wrap-fix で有効化
        enable_red_hue_wrap_fix=args.enable_red_hue_wrap_fix,
        # 案D 光沢ハイライト除外彩度計算 (fix/v70-zeropatch-redyellow): --specular-robust-saturation で有効化
        enable_specular_robust_saturation=args.enable_specular_robust_saturation,
        # 設計C 事後復旧ゲート (2026-06-02): --stable-recovery-gate で有効化
        enable_stable_recovery_gate=args.enable_stable_recovery_gate,
        # フェーズ A4 (2026-06-02): --enable-ojama-visual-detection で有効化
        enable_ojama_visual_detection=args.enable_ojama_visual_detection,
        # フェーズ A4 (2026-06-02): --enable-ojama-visual-chain-exit で有効化
        enable_ojama_visual_chain_exit=args.enable_ojama_visual_chain_exit,
        # フェーズ A4 (2026-06-02): --enable-ojama-infer-guard で有効化
        enable_ojama_infer_guard=args.enable_ojama_infer_guard,
        # フェーズ A4 (2026-06-02): --enable-ojama-settle-detection で有効化
        enable_ojama_settle_detection=args.enable_ojama_settle_detection,
        # 機能B (2026-06-02): --chain-score-early-fire で有効化
        enable_chain_score_early_fire=args.enable_chain_score_early_fire,
        # 機能C (2026-06-02): --chain-exit-warmup で有効化
        enable_chain_exit_warmup=args.enable_chain_exit_warmup,
        # 機能D (2026-06-02): --chain-formula-detection で有効化
        enable_chain_formula_detection=args.enable_chain_formula_detection,
        # 案 Y-4 (2026-06-03): --hsv-deferred-consensus で有効化
        enable_hsv_deferred_consensus=args.enable_hsv_deferred_consensus,
        # 不具合B 対処 (2026-06-04): --ojama-warning-glow-guard で有効化
        enable_ojama_warning_glow_guard=args.enable_ojama_warning_glow_guard,
        # 案P3 (2026-06-05): --chain-max-hold-override で有効化
        enable_chain_max_hold_override=args.enable_chain_max_hold_override,
        # 案X*(A)(B)+warmup (2026-06-05): --chain-exit-next-signal で有効化
        enable_chain_exit_next_signal=args.enable_chain_exit_next_signal,
        # feat/gravity-settle-2026-06-05: --gravity-settle-state で有効化
        enable_gravity_settle_state=args.enable_gravity_settle_state,
        # 案γ (2026-06-06): --slide-override-ojama-hold で有効化
        enable_slide_override_ojama_hold=args.enable_slide_override_ojama_hold,
    )
    if args.patch_ncc_threshold is not None:
        print(f"[viz] patch_ncc_threshold={args.patch_ncc_threshold} (NCC sweep)")
    # 案 R3 改 (2026-05-28): pipeline に video_id を設定
    # bg_fp 採取完了後に per-video PuyoColorProfileDB が自動ロードされる
    _vid_match = __import__("re").search(r"(v\d+)", args.video.name)
    if _vid_match and hasattr(pipeline, "set_video_id"):
        _vid_id = _vid_match.group(1)
        pipeline.set_video_id(_vid_id)
        print(f"[viz] puyo_profile video_id={_vid_id} (R3改: per-video profile 自動ロード)")
    if args.enable_warmup_guard:
        print("[viz] enable_warmup_guard=ON (B1: STABLE 直後 confirmed 凍結)")
    if args.enable_piece_persistence:
        print("[viz] enable_piece_persistence=ON (B1: STABLE 中 cell 色保護 / 散発色ブレ削減)")
    if args.enable_tier1_warmup:
        print(
            "[viz] enable_tier1_warmup=ON "
            "(NON-STABLE→STABLE 遷移直後 3 frame tier1 skip / 着地直後誤 EMPTY 化防止)"
        )
    if args.enable_ojama_tier1_warmup:
        print(
            "[viz] enable_ojama_tier1_warmup=ON "
            "(経路 A': OJAMA_FALL→STABLE 遷移直後 8 frame tier1 skip / v70 列崩壊対策)"
        )
    if not args.enable_constraint_fill:
        print("[viz] constraint_fill=OFF (案2: constraint_fill 無効 / CNN 高確信セル保護)")
    elif args.enable_constraint_fill:
        print("[viz] constraint_fill=ON (明示指定: constraint_fill 有効化)")
    if args.enable_t2_highconf_yield:
        print(
            "[viz] t2_highconf_yield=ON "
            "(T2 高確信 yield: CNN 支持セルは prev_stable 上書きスキップ / "
            "infer_placement 誤推論 + T2 自己強化フリーズ修正)"
        )
    if args.enable_infer_empty_guard:
        print(
            "[viz] infer_empty_guard=ON "
            "(空セル hallucination ガード: 非 diff セルが CNN EMPTY なら候補スキップ)"
        )
    if args.enable_game_event_chain_exit:
        print(
            "[viz] game_event_chain_exit=ON "
            "(game-event ベース連鎖終了: 次ツモ変化 / お邪魔降下で CHAIN 終了 / "
            f"安全弁 max={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s)"
        )
    if args.enable_landing_color_fix:
        print(
            "[viz] landing_color_fix=ON "
            "(着地色修正 案1: falling_pair を _landing_pending 消費色に切り替え / "
            "slide_motion 経由の 1 つ前ツモ色誤書き修正)"
        )
    if args.enable_chain_min_display:
        print(
            "[viz] chain_min_display=ON "
            f"(X1: 最小{RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s 表示保証 / "
            f"X4: chain_count < {RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT} で exit 抑止)"
        )
    if args.enable_hsv_classify_fallback:
        print(
            "[viz] hsv_classify_fallback=ON "
            "(2 択強制確定回避: 両候補拮抗/遠い/低彩度で next_pair 素返し / "
            "黄→赤誤分類 ~900 件発火点対策)"
        )
    if args.enable_landing_observed_color:
        print(
            "[viz] landing_observed_color=ON "
            "(真因 A 対処: 着地 2 cell の CNN==HSV 一致色で falling_pair ズレを補正)"
        )
    # フェーズ A4 (2026-06-02): お邪魔ぷよ視覚的検出関連ログ
    if args.enable_ojama_visual_detection:
        print("[viz] enable_ojama_visual_detection=ON (フェーズ A4: お邪魔ぷよ視覚的検出)")
    if args.enable_ojama_visual_chain_exit:
        print("[viz] enable_ojama_visual_chain_exit=ON (フェーズ A4: お邪魔ぷよ視覚的連鎖終了判定)")
    if args.enable_ojama_infer_guard:
        print("[viz] enable_ojama_infer_guard=ON (フェーズ A4: お邪魔ぷよ推論ガード)")
    if args.enable_ojama_settle_detection:
        print("[viz] enable_ojama_settle_detection=ON (フェーズ A4: お邪魔ぷよ着地検出)")
    if args.enable_chain_score_early_fire:
        print(
            "[viz] chain_score_early_fire=ON "
            f"(機能B: score >= {80} で即 CHAIN 突入 / VideoChainTracker フォールバック維持)"
        )
    if args.enable_chain_exit_warmup:
        print(
            "[viz] chain_exit_warmup=ON "
            f"(機能C: CHAIN→STABLE 後 {0.1}s confirmed 凍結 / エフェクト残光混入防止)"
        )
    if args.enable_chain_max_hold_override:
        print(
            "[viz] chain_max_hold_override=ON "
            f"(案P3: CHAIN_MAX_HOLD_SEC={RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s 超過後 "
            "ojama 保留を強制解除 / 連鎖過剰保持 v89 t34-40.87 修正)"
        )
    if args.enable_chain_exit_next_signal:
        print(
            "[viz] chain_exit_next_signal=ON "
            "(案X*: (A) 機能D CHAIN 中再点火抑制 + (B) NextSlide で CHAIN 即終了 "
            f"+ warmup {0.1}s confirmed 凍結 / v89 1P 6.87s 過剰保持根本修正)"
        )
    if args.enable_gravity_settle_state:
        from src.board_state_machine import (
            GRAVITY_SETTLE_MIN_FRAMES, GRAVITY_SETTLE_MAX_SEC,
            GRAVITY_SETTLE_PHYSICS_CLEAR_MIN,
        )
        print(
            "[viz] gravity_settle_state=ON "
            f"(CHAIN → GRAVITY_SETTLE → STABLE: "
            f"min={GRAVITY_SETTLE_MIN_FRAMES}f physics_clear={GRAVITY_SETTLE_PHYSICS_CLEAR_MIN}f "
            f"timeout={GRAVITY_SETTLE_MAX_SEC}s / 連鎖後 settle 採点外)"
        )
    if args.enable_slide_override_ojama_hold:
        print(
            "[viz] slide_override_ojama_hold=ON "
            "(案γ: CHAIN 中 slide_motion=True が ojama-hold ガードを上書きして CHAIN 終了 "
            "/ v89 t35.2-39.67 連鎖過剰保持修正、2026-06-06 採用)"
        )
    else:
        print("[viz] slide_override_ojama_hold=OFF (--no-slide-override-ojama-hold 指定)")
    if args.bg_fp_force_max_puyo is not None:
        print(f"[viz] bg_fp_force_max_puyo={args.bg_fp_force_max_puyo} (B2: FP 採取制限)")
    # Step 0 (2026-05-24): --no-online-hsv で OnlineHsvCalibrator を無効化
    if args.no_online_hsv:
        pipeline._online_hsv = None
        print("[viz] online_hsv DISABLED (= Step 0 比較用)")
    if args.vote_mode:
        print("[viz] vote_mode=ON (per-pixel HSV voting)")
    if args.cnn_override_prob is not None:
        print(f"[viz] cnn_override_prob={args.cnn_override_prob}")
    if args.mask_ojama_logit:
        print("[viz] mask_ojama_logit=ON (cycle 32e)")
    if args.use_puyo_gate:
        print("[viz] use_puyo_gate=ON (cycle 32e)")
    # 2026-05-11 サイクル63: 元動画解像度を image_reader に通知
    # (image_reader.read_both_boards で 1920x1080 にリサイズされるため、
    # pipeline.update に渡る frame からは元解像度が分からない)。
    if hasattr(pipeline._reader, "set_resolution_aware_s_min"):
        pipeline._reader.set_resolution_aware_s_min(height)
        print(f"[viz] resolution-aware S_min applied for source height={height}")
    # cycle 71r (案 A, 2026-05-13): BoardRegion 自動 calibration.
    # cycle 71u (2026-05-13 副作用対策): 案 A を撤回. 誤った座標補正で
    # ベース認識精度が悪化 (= 「双方とも認識悪化」 ユーザー報告) のため.
    # 必要なら --auto-calibrate 引数で明示的に有効化する.
    # 巻き戻し不要 (= cap は最初から再開).
    pass
    # サイクル4: 動画別 HSV ranges DB を起動時に inject
    if args.hsv_state is not None:
        try:
            import json as _json
            with args.hsv_state.open("r", encoding="utf-8") as _f:
                _state = _json.load(_f)
            _ranges = _state.get("per_video_ranges", {})
            _ranges_int = {
                int(k): tuple(int(x) for x in v) for k, v in _ranges.items()
            }
            from src.hybrid_classifier import HybridClassifier
            _hc = pipeline._reader._classifier
            if (
                isinstance(_hc, HybridClassifier)
                and hasattr(_hc._hsv, "set_color_ranges_from_simple")
                and _ranges_int
            ):
                _hc._hsv.set_color_ranges_from_simple(_ranges_int)
                # 循環 Hue 補完 guard: 赤等の2範囲定義色で per_video inject が
                # 片側 (H=0-13) を欠落させていないか確認し不足分を追加する。
                _ensure_circular_ranges_guard(_hc._hsv)
                # 2026-05-11 サイクル63 #6: 低解像度では pre-inject 後も
                # OnlineHsv の学習を継続させる (= merged_default は generic
                # で動画固有の調整が必要なため). 720p+ は DB が動画別で
                # tight なので従来通り suppress.
                if pipeline._online_hsv is not None and height >= 720:
                    pipeline._online_hsv_injected = True
                print(
                    f"[viz] HSV pre-inject from {args.hsv_state}: "
                    f"{len(_ranges_int)} colors "
                    f"(online_hsv {'suppressed' if height >= 720 else 'continues'})",
                )
        except Exception as _e:
            print(f"[viz] HSV pre-inject failed: {_e}", file=sys.stderr)

    sample_interval_frames = max(1, int(round(args.sample_interval * fps)))
    last_p1_state = BoardState.MENU
    last_p2_state = BoardState.MENU
    # 評価で使う盤面 = STABLE 時の confirmed_board を凍結保持
    # NON-STABLE (chain/tsumo_fall/ojama_fall/effect) では更新せず、前回 STABLE 値維持
    last_p1_eval_board: Board | None = None
    last_p2_eval_board: Board | None = None
    # cycle 33: board log JSONL 出力 (= 強化アナリスト用)
    board_log_fp = None
    if args.dump_board_log is not None:
        args.dump_board_log.parent.mkdir(parents=True, exist_ok=True)
        board_log_fp = open(args.dump_board_log, "w", encoding="utf-8")
        print(f"[viz] board log → {args.dump_board_log}")
    # Q1 全力施策: 詳細 board log JSONL (raw_hsv / bg_fp_distance / tier1 等)
    board_log_detail_fp = None
    if args.dump_board_log_detailed is not None:
        args.dump_board_log_detailed.parent.mkdir(parents=True, exist_ok=True)
        board_log_detail_fp = open(args.dump_board_log_detailed, "w", encoding="utf-8")
        print(f"[viz] detailed board log → {args.dump_board_log_detailed}")
    # 詳細 dump 用: prev frame の confirmed_board (physics_fix diff 計算用)
    prev_p1_confirmed: Board | None = None
    prev_p2_confirmed: Board | None = None

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        # 認識実行 (sample_interval_frames ごと)
        if fi % sample_interval_frames == 0:
            result = pipeline.update(fi, t_sec, frame)
            last_p1_state = result.p1.state
            last_p2_state = result.p2.state
            # STABLE 時のみ確定盤面を取得 (= indicator 評価で使うのと同じ条件)
            if (result.p1.state == BoardState.STABLE
                    and result.p1.confirmed_board is not None):
                last_p1_eval_board = result.p1.confirmed_board
            if (result.p2.state == BoardState.STABLE
                    and result.p2.confirmed_board is not None):
                last_p2_eval_board = result.p2.confirmed_board
            # cycle 33: 各 frame の認識結果を JSONL に保存
            if board_log_fp is not None:
                import json as _json
                # erasure_alerts: SideResult に乗った「当該 frame での新規 alert」
                # backwards compat: キーなし既存 JSONL を読む側は .get('p1_erasure_alerts', []) で対処。
                p1_ea = result.p1.erasure_alerts if result.p1.erasure_alerts is not None else []
                p2_ea = result.p2.erasure_alerts if result.p2.erasure_alerts is not None else []
                entry = {
                    "frame_idx": fi,
                    "t_sec": t_sec,
                    "p1_state": result.p1.state.value,
                    "p2_state": result.p2.state.value,
                    "p1_confirmed": (
                        result.p1.confirmed_board.to_dict()["grid"]
                        if result.p1.confirmed_board is not None else None
                    ),
                    "p2_confirmed": (
                        result.p2.confirmed_board.to_dict()["grid"]
                        if result.p2.confirmed_board is not None else None
                    ),
                    # T4 PuyoErasureMonitor: 当該 frame で検出した新規 alert [(row, col), ...]
                    "p1_erasure_alerts": [list(a) for a in p1_ea],
                    "p2_erasure_alerts": [list(a) for a in p2_ea],
                }
                board_log_fp.write(_json.dumps(entry, ensure_ascii=False) + "\n")
            # Q1 全力施策: 詳細 JSONL 出力 (raw_hsv / bg_fp_distance 等)
            if board_log_detail_fp is not None:
                import json as _json2
                try:
                    detail_entry = _build_detailed_log_entry(
                        fi, t_sec, result, frame, pipeline,
                        prev_p1_confirmed, prev_p2_confirmed,
                    )
                except Exception as _detail_err:
                    # 詳細 dump 失敗は本流に影響させない
                    detail_entry = {
                        "frame_idx": fi, "t_sec": t_sec,
                        "error": str(_detail_err),
                    }
                board_log_detail_fp.write(
                    _json2.dumps(detail_entry, ensure_ascii=False) + "\n"
                )
            # prev_confirmed を更新 (次 frame の physics_fix diff 計算用)
            prev_p1_confirmed = (
                result.p1.confirmed_board.copy()
                if result.p1.confirmed_board is not None else prev_p1_confirmed
            )
            prev_p2_confirmed = (
                result.p2.confirmed_board.copy()
                if result.p2.confirmed_board is not None else prev_p2_confirmed
            )
        # 描画用エイリアス
        last_p1_board = last_p1_eval_board
        last_p2_board = last_p2_eval_board

        # 描画
        draw_cell_overlay(frame, last_p1_board, P1_ROI_X, P1_ROI_Y)
        draw_cell_overlay(frame, last_p2_board, P2_ROI_X, P2_ROI_Y)
        draw_state_label(frame, last_p1_state, P1_ROI_X, P1_ROI_Y, label_prefix="1P:")
        draw_state_label(frame, last_p2_state, P2_ROI_X, P2_ROI_Y, label_prefix="2P:")
        draw_global_info(frame, fi, t_sec, last_p1_state, last_p2_state)

        writer.write(frame)
        if fi % 100 == 0:
            print(f"  [progress] {fi}/{n_frames} ({fi*100/max(n_frames,1):.1f}%) "
                  f"1P={last_p1_state.value} 2P={last_p2_state.value}")

    cap.release()
    writer.release()
    if board_log_fp is not None:
        board_log_fp.close()
        print(f"[done] board log saved")
    if board_log_detail_fp is not None:
        board_log_detail_fp.close()
        print(f"[done] detailed board log saved → {args.dump_board_log_detailed}")
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
