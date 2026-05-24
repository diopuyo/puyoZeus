"""フィールド読取の誤認識を診断する (Phase T サイクル 7)。

特定フレームの 1P/2P 全 72 セル (12×6) について:
    - パッチ画像 (元画像、拡大表示)
    - HSV 中央値 (h, s, v)
    - 各色の HSV 範囲適合 (赤/青/緑/黄/紫 のいずれか)
    - 最終分類結果
    - empty 判定理由 (V<EMPTY_V_THRESHOLD or HSV 範囲不一致 etc)

を grid 画像にまとめて出力。

ユーザは grid を見て:
    - 「ここに紫ぷよがあるはずだが empty 判定 (理由: V=45 で暗所紫が空に倒れた)」
    - 「ここは ROI が cell 中央からズレていてキャラ画像を拾っている」
    などを目視で特定可能。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.image_reader import (
    DEFAULT_COLOR_RANGES,
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    EMPTY_V_THRESHOLD,
    ImageReader,
)

COLOR_NAME = {
    COLOR_EMPTY: "empty",
    COLOR_RED: "red",
    COLOR_BLUE: "blue",
    COLOR_GREEN: "green",
    COLOR_YELLOW: "yellow",
    COLOR_PURPLE: "purple",
    9: "ojama",
    10: "?",
}
COLOR_BGR_DRAW = {
    COLOR_EMPTY: (40, 40, 40),
    COLOR_RED: (60, 60, 220),
    COLOR_BLUE: (220, 100, 60),
    COLOR_GREEN: (60, 200, 60),
    COLOR_YELLOW: (60, 220, 220),
    COLOR_PURPLE: (200, 60, 200),
    9: (180, 180, 180),
    10: (80, 80, 120),
}


def classify_with_diagnosis(patch: np.ndarray) -> tuple[int, dict]:
    """ImageReader.ColorClassifier の挙動を診断付きで再現。"""
    diag: dict = {}
    if patch.size == 0:
        return COLOR_EMPTY, {"reason": "empty_patch"}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = int(np.median(hsv[:, :, 0]))
    s = int(np.median(hsv[:, :, 1]))
    v = int(np.median(hsv[:, :, 2]))
    diag["h"] = h
    diag["s"] = s
    diag["v"] = v

    if v < EMPTY_V_THRESHOLD:
        diag["reason"] = f"V<{EMPTY_V_THRESHOLD} (V={v})"
        return COLOR_EMPTY, diag

    matches: list[tuple[int, str]] = []
    for color_id, ranges in DEFAULT_COLOR_RANGES.items():
        for rng in ranges:
            in_h = (
                rng.h_min <= h <= rng.h_max
                if rng.h_min <= rng.h_max
                else (h >= rng.h_min or h <= rng.h_max)
            )
            in_s = rng.s_min <= s <= getattr(rng, "s_max", 255)
            in_v = rng.v_min <= v <= getattr(rng, "v_max", 255)
            if in_h and in_s and in_v:
                matches.append((color_id, COLOR_NAME[color_id]))
                break
    diag["matches"] = [m[1] for m in matches]
    if matches:
        return matches[0][0], diag
    diag["reason"] = (
        f"no HSV match (h={h}, s={s}, v={v}); "
        f"thresholds: red s>=120 v>=100, blue s>=100 v>=80, "
        f"green s>=100 v>=80, yellow s>=100 v>=100, purple s>=80 v>=80"
    )
    return COLOR_EMPTY, diag


def render_cell_panel(
    patch: np.ndarray,
    diag: dict,
    final_color: int,
    label: str,
    scale: int = 4,
) -> np.ndarray:
    """1 セル分のパネル: パッチ拡大 + HSV + 判定結果を表示。"""
    if patch.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    big = cv2.resize(
        patch, (patch.shape[1] * scale, patch.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    # 下に情報帯 60px
    info_h = 80
    info = np.zeros((info_h, big.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        info, label, (3, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 100), 1,
    )
    h = diag.get("h", 0)
    s = diag.get("s", 0)
    v = diag.get("v", 0)
    cv2.putText(
        info, f"H{h:3d} S{s:3d} V{v:3d}", (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
    )
    name = COLOR_NAME.get(final_color, "?")
    color_bgr = COLOR_BGR_DRAW.get(final_color, (100, 100, 100))
    cv2.putText(
        info, f"=> {name}", (3, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1,
    )
    reason = diag.get("reason", "")
    if reason:
        cv2.putText(
            info, reason[:30], (3, 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 100), 1,
        )
    return np.vstack([big, info])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--time", type=float, default=30.0,
                        help="動画内の解析時刻 (秒)")
    parser.add_argument("--out", type=str,
                        default="data/verify/diag_field_grid.png")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"動画オープン失敗: {args.video}")
        return 1
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("frame 取得失敗")
        return 1
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

    # 各セルを診断
    panels_1p: list[list[np.ndarray]] = []
    panels_2p: list[list[np.ndarray]] = []

    for r_visible in range(VISIBLE_ROWS):
        row_1p: list[np.ndarray] = []
        row_2p: list[np.ndarray] = []
        row = r_visible + HIDDEN_ROWS
        for col in range(BOARD_COLS):
            for region, panels in (
                (DEFAULT_P1_REGION, row_1p),
                (DEFAULT_P2_REGION, row_2p),
            ):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                patch = frame[y1:y2, x1:x2]
                color, diag = classify_with_diagnosis(patch)
                panel = render_cell_panel(
                    patch, diag, color,
                    f"R{r_visible}C{col}",
                )
                panels.append(panel)
        panels_1p.append(row_1p)
        panels_2p.append(row_2p)

    # 1P と 2P を横に並べる
    def assemble(panels_grid: list[list[np.ndarray]], title: str) -> np.ndarray:
        # 各 row を横に結合、全 row を縦結合
        rows: list[np.ndarray] = []
        for row in panels_grid:
            sep = np.full((row[0].shape[0], 4, 3), 60, dtype=np.uint8)
            parts: list[np.ndarray] = []
            for p in row:
                parts.append(p); parts.append(sep)
            rows.append(np.hstack(parts[:-1]))
        sep_v = np.full((6, rows[0].shape[1], 3), 30, dtype=np.uint8)
        parts2: list[np.ndarray] = []
        for r in rows:
            parts2.append(r); parts2.append(sep_v)
        body = np.vstack(parts2[:-1])
        # ヘッダ
        header = np.zeros((40, body.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header, title, (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 100), 2,
        )
        return np.vstack([header, body])

    grid_1p = assemble(panels_1p, "1P")
    grid_2p = assemble(panels_2p, "2P")
    sep = np.full((20, max(grid_1p.shape[1], grid_2p.shape[1]), 3),
                  20, dtype=np.uint8)
    # 高さ・幅揃え
    if grid_1p.shape[1] < grid_2p.shape[1]:
        pad = np.full(
            (grid_1p.shape[0], grid_2p.shape[1] - grid_1p.shape[1], 3),
            20, dtype=np.uint8,
        )
        grid_1p = np.hstack([grid_1p, pad])
    elif grid_2p.shape[1] < grid_1p.shape[1]:
        pad = np.full(
            (grid_2p.shape[0], grid_1p.shape[1] - grid_2p.shape[1], 3),
            20, dtype=np.uint8,
        )
        grid_2p = np.hstack([grid_2p, pad])

    grid = np.vstack([grid_1p, sep, grid_2p])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"出力: {out_path} (shape={grid.shape})")
    print(f"\n元動画 t={args.time}s のフレームの全 72×2 セル診断結果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
