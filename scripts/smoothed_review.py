"""
動画の特定時刻を中心に ±0.25 秒 (計 0.5 秒) の連続フレームを抽出し、
per-cell 最頻色で安定化した盤面を color_review 形式で出力する。

UI/halo/連鎖エフェクトなど「1 フレームだけ現れるノイズ」を多数決で除去する。

使い方:
    ./venv/bin/python scripts/smoothed_review.py \\
        --video data/frames/video_01.mp4 \\
        --at 30 60 90 \\
        --stride 2

出力:
    data/verify/smoothed_review_<video_stem>_t<tss>.png

CNN は cnn_global_best.pt を優先使用。
学習への影響回避: CUDA_VISIBLE_DEVICES="" で CPU 推論。
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
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.physics_sanity import PhysicsSanityChecker, ViolationKind
from src.stateful_board_tracker import StatefulBoardTracker
from src.temporal_smoother import TemporalSmoother

# ============================
# 可視化定数 (verify_color_classification と同等)
# ============================

CLASS_DISPLAY: dict[int, tuple[str, tuple[int, int, int]]] = {
    COLOR_EMPTY: ("空", (64, 64, 64)),
    COLOR_RED: ("赤", (0, 0, 220)),
    COLOR_BLUE: ("青", (220, 80, 0)),
    COLOR_GREEN: ("緑", (0, 180, 0)),
    COLOR_YELLOW: ("黄", (0, 200, 220)),
    COLOR_PURPLE: ("紫", (180, 0, 180)),
    COLOR_OJAMA: ("お", (200, 200, 200)),
}

TILE_SIZE: int = 48
LABEL_HEIGHT: int = 16
SIDE_GAP: int = 32
OUTPUT_DIR: Path = Path("data/verify")

# ============================
# CNN モデルパス
# ============================

_GLOBAL_BEST: Path = Path("models/cnn_global_best.pt")
_LATEST: Path = Path("models/cnn_best.pt")
DEFAULT_CNN: Path = _GLOBAL_BEST if _GLOBAL_BEST.exists() else _LATEST
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")


def _extract_frames_around(
    video_path: Path,
    center_sec: float,
    window_sec: float,
    stride: int,
) -> list[np.ndarray]:
    """
    動画の center_sec を中心に ±window_sec/2 秒ぶんのフレームを抽出する。

    stride 指定で、動画 fps に対して何フレームおきにサンプルするかを決める。
    例: 動画が 60fps、stride=2 なら実効 30fps サンプル。

    Returns:
        BGR 画像リスト (古い順)。動画が短い/範囲外なら空リストを返す。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            return []

        half = window_sec / 2.0
        start_sec = max(0.0, center_sec - half)
        end_sec = min(total_frames / fps, center_sec + half)

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        frames: list[np.ndarray] = []
        for idx in range(start_frame, end_frame, max(1, stride)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[:2] != (1080, 1920):
                # 1920×1080 以外はキャリブレーション不整合なのでスキップ
                continue
            frames.append(frame)
        return frames
    finally:
        cap.release()


def _classify_board(
    frame: np.ndarray,
    region: BoardRegion,
    gated: GatedCnnClassifier,
) -> Board:
    """1 フレーム 1 側を CNN で分類して Board にする。隠し段は empty 固定。"""
    board = Board()
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            patch = frame[y1c:y2c, x1c:x2c]
            board.set(row, col, gated.classify(patch))
    return board


def _render_tile(
    patch: np.ndarray,
    label: int,
    violation_kinds: set[ViolationKind] | None = None,
) -> np.ndarray:
    """verify_color_classification と同等のタイル描画。"""
    tile = np.zeros((TILE_SIZE + LABEL_HEIGHT, TILE_SIZE, 3), dtype=np.uint8)
    if patch is not None and patch.size > 0:
        resized = cv2.resize(patch, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
        tile[:TILE_SIZE, :, :] = resized
    name, color = CLASS_DISPLAY.get(label, ("?", (128, 128, 128)))
    tile[TILE_SIZE:, :, :] = color
    cv2.putText(
        tile, name, (4, TILE_SIZE + LABEL_HEIGHT - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
        1, cv2.LINE_AA,
    )
    if violation_kinds:
        cv2.rectangle(tile, (0, 0), (TILE_SIZE - 1, TILE_SIZE + LABEL_HEIGHT - 1),
                      (0, 0, 255), 3)
        markers = []
        if ViolationKind.AIRBORNE in violation_kinds:
            markers.append("浮")
        if ViolationKind.UNRESOLVED_CHAIN in violation_kinds:
            markers.append("4+")
        if markers:
            cv2.putText(tile, "/".join(markers), (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return tile


def _compose_side(
    board: Board,
    patches: list[list[np.ndarray]],
    side_name: str,
    violations_by_cell: dict[tuple[int, int], set[ViolationKind]],
    title_suffix: str,
) -> np.ndarray:
    """1 側 (1P/2P) のグリッド画像を組み立てる。"""
    tile_h = TILE_SIZE + LABEL_HEIGHT
    header_h = 44
    grid_w = BOARD_COLS * TILE_SIZE
    grid_h = header_h + BOARD_ROWS * tile_h
    canvas = np.full((grid_h, grid_w, 3), 32, dtype=np.uint8)

    cv2.putText(canvas, f"{side_name} {title_suffix}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    n_air = sum(1 for v in violations_by_cell.values() if ViolationKind.AIRBORNE in v)
    n_chain = sum(1 for v in violations_by_cell.values() if ViolationKind.UNRESOLVED_CHAIN in v)
    cv2.putText(canvas, f"violations: 浮遊={n_air} 4+={n_chain}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            label = board.get(row, col)
            patch = patches[row][col]
            viol = violations_by_cell.get((row, col))
            tile = _render_tile(patch, label, viol)
            y0 = header_h + row * tile_h
            x0 = col * TILE_SIZE
            canvas[y0:y0 + tile_h, x0:x0 + TILE_SIZE, :] = tile
    return canvas


def _extract_patches(frame: np.ndarray, region: BoardRegion) -> list[list[np.ndarray]]:
    """中央フレームのパッチ画像を row×col 2D リストで返す (表示用)。"""
    patches: list[list[np.ndarray]] = []
    for row in range(BOARD_ROWS):
        row_patches: list[np.ndarray] = []
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                row_patches.append(np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8))
            else:
                row_patches.append(frame[y1c:y2c, x1c:x2c].copy())
        patches.append(row_patches)
    return patches


def _format_counts(board: Board) -> str:
    counts: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = board.get(r, c)
            counts[v] = counts.get(v, 0) + 1
    parts = []
    for cls in (COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
                COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA):
        parts.append(f"{CLASS_DISPLAY[cls][0]}={counts.get(cls, 0)}")
    return " ".join(parts)


def process_time(
    video_path: Path,
    center_sec: float,
    window_sec: float,
    stride: int,
    gated: GatedCnnClassifier,
    p1_region: BoardRegion,
    p2_region: BoardRegion,
    out_dir: Path,
    prerolled_sec: float = 1.0,
) -> Path | None:
    """指定時刻の安定化レビュー画像を生成する。

    StatefulBoardTracker を安定化するため、center_sec より
    prerolled_sec 秒ぶん前からフレームを処理し、tracker を温めてから
    [center_sec - window_sec/2, center_sec + window_sec/2] の安定状態を出力する。
    """
    # preroll + 観測窓の全域を抽出
    full_window = prerolled_sec + window_sec
    total_center = center_sec - prerolled_sec / 2.0 + window_sec / 2.0
    frames = _extract_frames_around(video_path, total_center, full_window, stride)
    if len(frames) < 3:
        print(f"  t={center_sec:.1f}s: フレーム不足 ({len(frames)}) スキップ")
        return None

    # 中央フレームを表示用に採用
    mid_frame = frames[len(frames) // 2]

    # 1P/2P 側で smoother (0.5秒窓) + stateful tracker
    smoother_window = max(3, int(window_sec / full_window * len(frames)))
    sm_1p = TemporalSmoother(window_size=smoother_window)
    sm_2p = TemporalSmoother(window_size=smoother_window)
    tr_1p = StatefulBoardTracker()
    tr_2p = StatefulBoardTracker()
    stable_1p = Board()
    stable_2p = Board()
    for f in frames:
        b1 = _classify_board(f, p1_region, gated)
        b2 = _classify_board(f, p2_region, gated)
        smooth_1p = sm_1p.update(b1)
        smooth_2p = sm_2p.update(b2)
        stable_1p = tr_1p.update(smooth_1p)
        stable_2p = tr_2p.update(smooth_2p)

    # 物理サニティ
    checker = PhysicsSanityChecker()
    viol_1p_list = checker.check(stable_1p)
    viol_2p_list = checker.check(stable_2p)
    viol_1p: dict[tuple[int, int], set[ViolationKind]] = {}
    viol_2p: dict[tuple[int, int], set[ViolationKind]] = {}
    for v in viol_1p_list:
        viol_1p.setdefault((v.row, v.col), set()).add(v.kind)
    for v in viol_2p_list:
        viol_2p.setdefault((v.row, v.col), set()).add(v.kind)

    # 表示用パッチは中央フレーム
    patches_1p = _extract_patches(mid_frame, p1_region)
    patches_2p = _extract_patches(mid_frame, p2_region)

    # 統計表示: tracker の受理/棄却数 (直近 update の)
    reject_str = ""
    if tr_1p.last_stats is not None:
        reject_str = f" rejected={tr_1p.last_stats.rejected}/{tr_2p.last_stats.rejected if tr_2p.last_stats else 0}"
    suffix = f"(smooth+stateful {len(frames)}f{reject_str})"
    grid_1p = _compose_side(stable_1p, patches_1p, "1P", viol_1p, suffix)
    grid_2p = _compose_side(stable_2p, patches_2p, "2P", viol_2p, suffix)
    gap = np.full((grid_1p.shape[0], SIDE_GAP, 3), 32, dtype=np.uint8)
    combined = np.hstack([grid_1p, gap, grid_2p])

    out_dir.mkdir(parents=True, exist_ok=True)
    tstr = f"{int(center_sec):04d}"
    out_path = out_dir / f"smoothed_review_{video_path.stem}_t{tstr}s.png"
    cv2.imwrite(str(out_path), combined)

    print(f"  t={center_sec:.1f}s ({len(frames)}f smoothing):")
    print(f"    1P: {_format_counts(stable_1p)}  違反={len(viol_1p_list)}")
    print(f"    2P: {_format_counts(stable_2p)}  違反={len(viol_2p_list)}")
    print(f"    → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True,
                        help="対象 mp4 (1920x1080 推奨)")
    parser.add_argument("--at", type=float, nargs="+", required=True,
                        help="対象時刻 (秒)、スペース区切りで複数指定")
    parser.add_argument("--window-sec", type=float, default=0.5,
                        help="スムージング窓 (秒)、デフォルト 0.5")
    parser.add_argument("--stride", type=int, default=2,
                        help="フレームサンプリング間隔、デフォルト 2 (60fps → 30fps)")
    parser.add_argument("--cnn", type=Path, default=DEFAULT_CNN)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"動画が見つかりません: {args.video}")
        sys.exit(1)

    print(f"CNN: {args.cnn}")
    print(f"動画: {args.video}")
    print(f"窓: {args.window_sec}s, stride: {args.stride}")
    print(f"対象時刻: {args.at}")

    cnn = CnnPatchClassifier.load(args.cnn)
    config = CalibratedConfig.load(str(args.calib))
    gated = GatedCnnClassifier(color_classifier=cnn)

    for t in args.at:
        process_time(
            args.video, t, args.window_sec, args.stride,
            gated, config.p1_region, config.p2_region, args.out_dir,
        )


if __name__ == "__main__":
    main()
