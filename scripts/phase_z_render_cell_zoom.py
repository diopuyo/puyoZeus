"""Phase Z violations を 1 PNG = 1 cell の zoom 比較画像で出力。

1 violation あたり下記レイアウトで 1 PNG を生成:

    +----------------------+--------------------+
    | header: id, side r,c, recognized, t      |
    +----------------------+--------------------+
    | zoom (該当 cell の   | full (1P+2P 全体  |
    |  周囲 5x5 高解像度,  |  縮小、該当 cell   |
    |  該当 cell に赤枠)   |  マーク)          |
    +----------------------+--------------------+

ファイル名は CSV の id 順に並ぶ形式:
    {id:04d}_{side}_r{row:02d}c{col}_t{time:07.2f}_rec_{recognized}.png

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_render_cell_zoom \
        --segment v06_m03_385_415
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.phase_z_extract_violations import is_hard_violation  # noqa: E402
from src.board import HIDDEN_ROWS  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
)

WEAK_ROOT = _ROOT / "data" / "verify" / "phase_z_review" / "weak_video_extra"
FRAME_GRID_MS = 500

# 描画色 (BGR)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BG = (24, 24, 24)

# zoom layout
ZOOM_HALF = 2  # 該当 cell から ±2 cells = 5x5 領域
ZOOM_OUT_W = 600  # zoom 領域出力幅
FULL_OUT_W = 800  # 全体縮小幅
HEADER_H = 60
SEPARATOR_PX = 6
PADDING = 12


def parse_segment_video(segment: str) -> str | None:
    match = re.match(r"v(\d+)_m\d+", segment)
    if not match:
        return None
    return f"video_{int(match.group(1)):02d}.mp4"


def collect_violations(labels_csv: Path) -> list[dict]:
    out: list[dict] = []
    with labels_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_chain") == "1":
                continue
            if not is_hard_violation(row.get("suspicious_reasons", "")):
                continue
            out.append({
                "id": int(row["id"]),
                "time": float(row["time"]),
                "side": row["side"],
                "row": int(row["row"]),
                "col": int(row["col"]),
                "recognized": row["recognized"],
            })
    return out


def cell_center_abs(side: str, row: int, col: int) -> tuple[int, int]:
    region: BoardRegion = (
        DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    )
    abs_row = row + HIDDEN_ROWS
    return region.cell_center(abs_row, col)


def cell_full_rect(
    side: str, row: int, col: int,
) -> tuple[int, int, int, int]:
    region: BoardRegion = (
        DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    )
    abs_row = row + HIDDEN_ROWS
    cx, cy = region.cell_center(abs_row, col)
    half_w = max(1, int(region.cell_width / 2))
    half_h = max(1, int(region.cell_height / 2))
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def crop_zoom(
    img: np.ndarray, side: str, row: int, col: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """該当 cell の周囲 (2*ZOOM_HALF + 1) x (2*ZOOM_HALF + 1) を切り出し。

    Returns:
        (cropped_image, cell_rect_in_crop)
    """
    region: BoardRegion = (
        DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    )
    cw = region.cell_width
    ch = region.cell_height
    cx, cy = cell_center_abs(side, row, col)
    pad_w = int(cw * (ZOOM_HALF + 0.5))
    pad_h = int(ch * (ZOOM_HALF + 0.5))
    h, w = img.shape[:2]
    x1 = max(0, cx - pad_w)
    x2 = min(w, cx + pad_w)
    y1 = max(0, cy - pad_h)
    y2 = min(h, cy + pad_h)
    crop = img[y1:y2, x1:x2].copy()
    # cell rect (full size) を crop 座標系に
    cell_x1, cell_y1, cell_x2, cell_y2 = cell_full_rect(side, row, col)
    rect_in_crop = (
        cell_x1 - x1, cell_y1 - y1,
        cell_x2 - x1, cell_y2 - y1,
    )
    return crop, rect_in_crop


def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == 0:
        return img
    scale = target_w / w
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_CUBIC)


def draw_rect_in_crop(
    img: np.ndarray, rect: tuple[int, int, int, int],
    scale_x: float, scale_y: float,
) -> None:
    x1, y1, x2, y2 = rect
    sx1 = int(x1 * scale_x)
    sy1 = int(y1 * scale_y)
    sx2 = int(x2 * scale_x)
    sy2 = int(y2 * scale_y)
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), RED, 4)


def draw_full_with_marker(
    frame: np.ndarray, vio: dict, target_w: int,
) -> np.ndarray:
    full = frame.copy()
    if full.shape[:2] != (1080, 1920):
        full = cv2.resize(
            full, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    x1, y1, x2, y2 = cell_full_rect(vio["side"], vio["row"], vio["col"])
    cv2.rectangle(full, (x1, y1), (x2, y2), RED, 5)
    return resize_keep_aspect(full, target_w)


def render_zoom_panel(
    frame: np.ndarray, vio: dict,
) -> np.ndarray:
    crop, rect = crop_zoom(frame, vio["side"], vio["row"], vio["col"])
    if crop.size == 0:
        return np.full((400, ZOOM_OUT_W, 3), BG, dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = ZOOM_OUT_W / w
    out = resize_keep_aspect(crop, ZOOM_OUT_W)
    draw_rect_in_crop(out, rect, scale, scale)
    return out


def compose_canvas(
    zoom: np.ndarray, full: np.ndarray, vio: dict,
) -> np.ndarray:
    body_h = max(zoom.shape[0], full.shape[0])
    total_w = zoom.shape[1] + SEPARATOR_PX + full.shape[1] + PADDING * 2
    total_h = HEADER_H + body_h + PADDING * 2
    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)
    # header
    header = (
        f"#{vio['id']}  {vio['side']} r{vio['row']:02d}c{vio['col']}  "
        f"recognized={vio['recognized']}  t={vio['time']:.2f}s"
    )
    cv2.rectangle(canvas, (0, 0), (total_w, HEADER_H), GRAY, -1)
    cv2.putText(
        canvas, header, (PADDING, HEADER_H - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "ZOOM (5x5 cells)", (PADDING, HEADER_H - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "FULL FRAME",
        (PADDING + zoom.shape[1] + SEPARATOR_PX, HEADER_H - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA,
    )
    # body 配置 (左: zoom、右: full)
    y0 = HEADER_H + PADDING
    x0 = PADDING
    canvas[y0:y0 + zoom.shape[0], x0:x0 + zoom.shape[1]] = zoom
    x1 = x0 + zoom.shape[1] + SEPARATOR_PX
    canvas[y0:y0 + full.shape[0], x1:x1 + full.shape[1]] = full
    return canvas


def load_frame(
    cap: cv2.VideoCapture, frames_dir: Path, time_sec: float,
) -> np.ndarray | None:
    ms_round = int(round(time_sec * 2) * FRAME_GRID_MS)
    fp = frames_dir / f"{ms_round:06d}.png"
    if fp.exists():
        return cv2.imread(str(fp))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def render_segment(segment: str) -> int:
    seg_dir = WEAK_ROOT / segment
    labels_csv = seg_dir / "labels.csv"
    frames_dir = seg_dir / "frames"
    out_dir = seg_dir / "violations_review" / "cell_zoom"

    if not labels_csv.exists():
        print(f"ERROR: {labels_csv} not found")
        return 1
    video_name = parse_segment_video(segment)
    if video_name is None:
        print(f"ERROR: cannot derive video name from {segment}")
        return 1
    video_path = _ROOT / "data" / "frames" / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    violations = collect_violations(labels_csv)
    if not violations:
        print(f"[zoom] {segment}: no hard violations")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    # 同 time の frame は cache して再利用
    frame_cache: dict[float, np.ndarray | None] = {}
    n_written = 0
    n_failed = 0
    for vio in sorted(violations, key=lambda v: v["id"]):
        t = vio["time"]
        if t not in frame_cache:
            frame_cache[t] = load_frame(cap, frames_dir, t)
        frame = frame_cache[t]
        if frame is None:
            n_failed += 1
            continue
        zoom = render_zoom_panel(frame, vio)
        full = draw_full_with_marker(frame, vio, FULL_OUT_W)
        canvas = compose_canvas(zoom, full, vio)
        fname = (
            f"{vio['id']:04d}_{vio['side']}_r{vio['row']:02d}"
            f"c{vio['col']}_t{vio['time']:07.2f}_rec_{vio['recognized']}.png"
        )
        cv2.imwrite(str(out_dir / fname), canvas)
        n_written += 1
    cap.release()

    print(
        f"[zoom] {segment}: {n_written} cells -> PNG"
        + (f" ({n_failed} failed)" if n_failed else "")
    )
    print(f"[zoom] saved: {to_windows_path(out_dir)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=str, required=True)
    args = parser.parse_args()
    return render_segment(args.segment)


if __name__ == "__main__":
    sys.exit(main())
