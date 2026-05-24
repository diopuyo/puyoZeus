"""STABLE state cell dataset 生成 (Phase B-7-1).

各動画を BoardStateMachine pipeline で回し、STABLE 確定 frame の各 cell
画像 + label (state machine が確定した色) を npz に出力する。

新方針 (project_recognition_strategy_pivot) に基づき、アクション中
(TSUMO_FALL/CHAIN/OJAMA_FALL/EFFECT) の cell は完全に除外する。

出力:
    data/training/stable_state/{video}_cells.npz
        images: (N, 40, 40, 3) BGR uint8
        labels: (N,) int8 (COLOR_EMPTY..COLOR_OJAMA, COLOR_UNKNOWN)
        side:   (N,) "1P" or "2P" str
        row:    (N,) int8
        col:    (N,) int8
        frame_idx: (N,) int32

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_collect_stable_dataset \
        --duration 30 --fps 5 --videos 1,5,10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, HIDDEN_ROWS, VISIBLE_ROWS, Board,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

CELL_PX: int = 40  # 出力 cell 画像のサイズ
DEFAULT_OUT_ROOT: Path = _ROOT / "data" / "training" / "stable_state"


def get_match1(video_id: int) -> tuple[float, float] | None:
    for ver in ("v5", "v4"):
        p = (
            _ROOT
            / f"data/verify/match_boundaries_{ver}/video_{video_id:02d}/"
            f"matches.tsv"
        )
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if len(rows) > 1:
            try:
                return float(rows[1][1]), float(rows[1][2])
            except (IndexError, ValueError):
                continue
    return None


def extract_cells(
    frame: np.ndarray, region: BoardRegion, board: Board,
) -> tuple[list[np.ndarray], list[int], list[int], list[int]]:
    """frame + region + board から cell 画像 + label のリストを返す.

    Returns:
        (cells, labels, rows, cols)
    """
    cells: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    for row in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch = cv2.resize(
                patch, (CELL_PX, CELL_PX), interpolation=cv2.INTER_AREA,
            )
            cells.append(patch)
            labels.append(int(board.get(row, col)))
            rows.append(row)
            cols.append(col)
    return cells, labels, rows, cols


def collect_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int,
    out_dir: Path,
) -> tuple[int, int]:
    """1 動画から STABLE state cell データを抽出し npz 保存.

    Returns:
        (n_stable_frames, n_cells_total)
    """
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return 0, 0

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[fail] v{video_id:02d}: video open failed")
        return 0, 0

    interval = 1.0 / fps_sample
    images: list[np.ndarray] = []
    labels: list[int] = []
    sides: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    frame_indices: list[int] = []

    last_stable_1p: bytes | None = None
    last_stable_2p: bytes | None = None
    n_stable_frames = 0

    t = start_sec
    frame_idx = 0
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)
        for side_label, side_res, last_stable in (
            ("1P", result.p1, last_stable_1p),
            ("2P", result.p2, last_stable_2p),
        ):
            if (
                side_res.state == BoardState.STABLE
                and side_res.confirmed_board is not None
            ):
                key = side_res.confirmed_board.to_json().encode("utf-8")
                if key == last_stable:
                    continue
                # 新しい STABLE 盤面 → cell 抽出
                region = (
                    DEFAULT_P1_REGION if side_label == "1P"
                    else DEFAULT_P2_REGION
                )
                cells, lbls, rs, cs = extract_cells(
                    frame, region, side_res.confirmed_board,
                )
                images.extend(cells)
                labels.extend(lbls)
                sides.extend([side_label] * len(cells))
                rows.extend(rs)
                cols.extend(cs)
                frame_indices.extend([frame_idx] * len(cells))
                n_stable_frames += 1
                if side_label == "1P":
                    last_stable_1p = key
                else:
                    last_stable_2p = key
        frame_idx += 1
        t += interval
    cap.release()

    if not images:
        print(f"[empty] v{video_id:02d}: no STABLE cells")
        return 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{video_id:02d}_cells.npz"
    np.savez_compressed(
        out_path,
        images=np.array(images, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int8),
        side=np.array(sides),
        row=np.array(rows, dtype=np.int8),
        col=np.array(cols, dtype=np.int8),
        frame_idx=np.array(frame_indices, dtype=np.int32),
    )
    print(
        f"[done] v{video_id:02d}: {n_stable_frames} STABLE frames, "
        f"{len(images)} cells -> {to_windows_path(out_path)}"
    )
    return n_stable_frames, len(images)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument(
        "--videos", type=str, default="",
        help="カンマ区切り (例: 1,5,10)。空なら全 v01-v19",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_ROOT,
    )
    args = parser.parse_args()

    if args.videos:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    else:
        target_ids = list(range(1, 20))

    total_frames = 0
    total_cells = 0
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            print(f"[skip] v{vid:02d}: no match boundary")
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        n_f, n_c = collect_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            out_dir=args.out_dir,
        )
        total_frames += n_f
        total_cells += n_c

    print(
        f"\n[summary] total: {total_frames} STABLE frames, "
        f"{total_cells} cells across {len(target_ids)} videos"
    )
    print(f"[summary] saved to: {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
