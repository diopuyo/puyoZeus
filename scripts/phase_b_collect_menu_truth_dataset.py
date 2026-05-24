"""メニュー画面 (試合外) を真値 EMPTY として hard negative 集める (Phase B-7-A2).

新方針 (project_recognition_strategy_pivot) に基づく訓練データ生成。
試合外 frame は盤面に puyo が **物理的に存在しない**ことが確定するので、
CNN が puyo (EMPTY 以外) と認識した cell は明確な false positive。
これを cell 画像 + EMPTY label で訓練データに採取する。

メリット:
    - 真値が物理的に確定 (循環学習にならない)
    - 実装簡単 (MatchStateDetector の判定をそのまま使う)
    - メニュー画面の偽陽性を直接削減

弱点:
    - 試合中の cell 認識精度を直接改善しない (= メニュー側の補正のみ)
    - MatchStateDetector が試合中を試合外と誤判定すると、訓練データが汚染
      → bg_value が確実に閾値を超える frame のみ採用する保険を入れる

出力: data/training/menu_truth/{video}_menu.npz
    images: (N, 40, 40, 3) BGR uint8
    labels: (N,) int8 (常に COLOR_EMPTY=0)
    cnn_pred: (N,) int8 (元の CNN 出力)
    side: (N,) "1P" or "2P"
    row, col: (N,) int8
    frame_idx: (N,) int32
    bg_value: (N,) float32 (採用判定時の HSV V 平均、再現用)
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
    BOARD_COLS, COLOR_EMPTY, HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion, ImageReader,
)
from src.match_state import MatchState, MatchStateDetector  # noqa: E402

CELL_PX: int = 16  # phase_u manual_labels.npz と同じ 16x16 で fine-tune に直接使える
DEFAULT_OUT_ROOT: Path = _ROOT / "data" / "training" / "menu_truth"

# bg_value がこれ以上で「確実にメニュー画面」と判定 (= 閾値より明るい)。
# match_state.py の IN_MATCH_V_MAX=170 より厳しめに設定し、訓練データ汚染を防ぐ。
SAFE_BG_VALUE_MIN: float = 200.0


def collect_video(
    video_id: int, fps_sample: float, max_seconds: float,
    out_dir: Path,
) -> tuple[int, int]:
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return 0, 0

    reader = ImageReader(use_match_state=False)
    detector = MatchStateDetector.load_default()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[fail] v{video_id:02d}")
        return 0, 0

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
        1.0, cap.get(cv2.CAP_PROP_FPS),
    )
    end_sec = min(duration, max_seconds)
    interval = 1.0 / fps_sample

    images: list[np.ndarray] = []
    labels: list[int] = []
    cnn_preds: list[int] = []
    sides: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    frame_indices: list[int] = []
    bg_vals: list[float] = []

    n_menu_frames = 0
    t = 0.0
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
        ms = detector.detect(frame)
        # 試合外 + bg_value が安全閾値以上なら採用
        if ms.state != MatchState.NOT_IN_MATCH:
            t += interval
            frame_idx += 1
            continue
        if ms.bg_value < SAFE_BG_VALUE_MIN:
            t += interval
            frame_idx += 1
            continue
        n_menu_frames += 1

        cnn_1p, cnn_2p = reader.read_both_boards(frame)
        for side_label, cnn_board, region in (
            ("1P", cnn_1p, DEFAULT_P1_REGION),
            ("2P", cnn_2p, DEFAULT_P2_REGION),
        ):
            for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
                for c in range(BOARD_COLS):
                    cnn_v = int(cnn_board.get(r, c))
                    if cnn_v == COLOR_EMPTY:
                        continue  # 既に正解、訓練データに不要
                    # hard negative: CNN が puyo と誤認した cell
                    x1, y1, x2, y2 = region.cell_sample_rect(r, c)
                    patch = frame[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    patch = cv2.resize(
                        patch, (CELL_PX, CELL_PX),
                        interpolation=cv2.INTER_AREA,
                    )
                    images.append(patch)
                    labels.append(COLOR_EMPTY)
                    cnn_preds.append(cnn_v)
                    sides.append(side_label)
                    rows.append(r)
                    cols.append(c)
                    frame_indices.append(frame_idx)
                    bg_vals.append(ms.bg_value)
        t += interval
        frame_idx += 1
    cap.release()

    if not images:
        print(
            f"[empty] v{video_id:02d}: {n_menu_frames} menu frames, "
            f"no false positives"
        )
        return n_menu_frames, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{video_id:02d}_menu.npz"
    np.savez_compressed(
        out_path,
        images=np.array(images, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int8),
        cnn_pred=np.array(cnn_preds, dtype=np.int8),
        side=np.array(sides),
        row=np.array(rows, dtype=np.int8),
        col=np.array(cols, dtype=np.int8),
        frame_idx=np.array(frame_indices, dtype=np.int32),
        bg_value=np.array(bg_vals, dtype=np.float32),
    )
    print(
        f"[done] v{video_id:02d}: {n_menu_frames} menu frames, "
        f"{len(images)} false positive cells "
        f"-> {to_windows_path(out_path)}"
    )
    return n_menu_frames, len(images)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-seconds", type=float, default=30.0,
        help="動画開始から何秒まで走査するか (試合前メニューは大抵 30 秒以内)",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_ROOT,
    )
    args = parser.parse_args()

    target_ids = (
        [int(s) for s in args.videos.split(",") if s.strip()]
        if args.videos else list(range(1, 20))
    )

    total_frames = 0
    total_cells = 0
    for vid in target_ids:
        nf, nc = collect_video(
            video_id=vid, fps_sample=args.fps,
            max_seconds=args.max_seconds, out_dir=args.out_dir,
        )
        total_frames += nf
        total_cells += nc

    print(
        f"\n[summary] {total_frames} menu frames, "
        f"{total_cells} false positive cells across {len(target_ids)} videos"
    )
    print(f"[summary] saved to: {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
