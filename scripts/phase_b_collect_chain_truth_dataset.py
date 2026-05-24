"""連鎖後盤面の物理推論真値で CNN hard negative を集める (Phase B-7-A).

新方針 (project_recognition_strategy_pivot) に基づく訓練データ生成。
B-7-1 (STABLE-only dataset) は label が CNN ベースで循環学習になるため、
本スクリプトでは **物理推論で真値が出せる場面** のみを訓練データの源とする。

仕組み:
    1. VideoChainTracker が ChainEvent を返した frame で連鎖発火を検出
    2. event.before_board (= 発火前安定盤面) を ChainSimulator.simulate
    3. 結果の final_board を「物理推論で確定した真値」として採用
    4. 同 frame の CNN 出力 cell と真値 cell が **不一致** = hard negative
    5. その cell 画像 + 真値 label を npz に追記

これで CNN を fine-tune すると、循環せず新情報 (= 物理推論で正解判明した
cell) を学習させられる。

出力: data/training/chain_truth/{video}_hard.npz
    images: (N, 40, 40, 3) BGR uint8
    labels: (N,) int8 (真値 = ChainSimulator final_board の cell 値)
    cnn_pred: (N,) int8 (元の CNN 出力、参考)
    side: (N,) "1P" or "2P"
    row: (N,) int8
    col: (N,) int8
    frame_idx: (N,) int32

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_collect_chain_truth_dataset \
        --duration 60 --fps 5 --videos 1,6
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

from src.board import BOARD_COLS, HIDDEN_ROWS, VISIBLE_ROWS  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

CELL_PX: int = 16  # phase_u manual_labels.npz と同じ 16x16 で fine-tune に直接使える
DEFAULT_OUT_ROOT: Path = _ROOT / "data" / "training" / "chain_truth"


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


def collect_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int, out_dir: Path,
) -> tuple[int, int, int]:
    """1 動画から chain_truth hard negative を抽出.

    Returns:
        (n_chain_events, n_hard_cells, n_total_cells_compared)
    """
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return 0, 0, 0

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[fail] v{video_id:02d}")
        return 0, 0, 0

    sim = ChainSimulator()
    images: list[np.ndarray] = []
    labels: list[int] = []
    cnn_preds: list[int] = []
    sides: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    frame_indices: list[int] = []

    n_chain_events = 0
    n_total_compared = 0

    last_event_id_1p = id(None)
    last_event_id_2p = id(None)

    interval = 1.0 / fps_sample
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

        for side_label, side_res, prev_id in (
            ("1P", result.p1, last_event_id_1p),
            ("2P", result.p2, last_event_id_2p),
        ):
            ev = side_res.chain_event
            if ev is None:
                continue
            ev_id = id(ev)
            if ev_id == prev_id:
                continue  # 同じ event を pipeline が hold 中、初出のみ処理
            # 新規 event = 連鎖発火 1 回ぶん
            n_chain_events += 1
            chain_result = sim.simulate(ev.before_board)
            if chain_result.chain_count == 0:
                if side_label == "1P":
                    last_event_id_1p = ev_id
                else:
                    last_event_id_2p = ev_id
                continue
            true_board = chain_result.final_board
            cnn_board = side_res.cnn_board
            region: BoardRegion = (
                DEFAULT_P1_REGION if side_label == "1P"
                else DEFAULT_P2_REGION
            )
            for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
                for c in range(BOARD_COLS):
                    n_total_compared += 1
                    true_v = int(true_board.get(r, c))
                    cnn_v = int(cnn_board.get(r, c))
                    if true_v == cnn_v:
                        continue
                    # hard negative: cell 画像 + 真値 label を抽出
                    x1, y1, x2, y2 = region.cell_sample_rect(r, c)
                    patch = frame[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    patch = cv2.resize(
                        patch, (CELL_PX, CELL_PX),
                        interpolation=cv2.INTER_AREA,
                    )
                    images.append(patch)
                    labels.append(true_v)
                    cnn_preds.append(cnn_v)
                    sides.append(side_label)
                    rows.append(r)
                    cols.append(c)
                    frame_indices.append(frame_idx)
            if side_label == "1P":
                last_event_id_1p = ev_id
            else:
                last_event_id_2p = ev_id
        frame_idx += 1
        t += interval
    cap.release()

    n_hard = len(images)
    if n_hard == 0:
        print(
            f"[empty] v{video_id:02d}: {n_chain_events} events, "
            f"no hard cells"
        )
        return n_chain_events, 0, n_total_compared

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{video_id:02d}_hard.npz"
    np.savez_compressed(
        out_path,
        images=np.array(images, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int8),
        cnn_pred=np.array(cnn_preds, dtype=np.int8),
        side=np.array(sides),
        row=np.array(rows, dtype=np.int8),
        col=np.array(cols, dtype=np.int8),
        frame_idx=np.array(frame_indices, dtype=np.int32),
    )
    hard_rate = (
        100.0 * n_hard / n_total_compared if n_total_compared else 0.0
    )
    print(
        f"[done] v{video_id:02d}: {n_chain_events} events, "
        f"{n_hard}/{n_total_compared} hard cells ({hard_rate:.1f}%) "
        f"-> {to_windows_path(out_path)}"
    )
    return n_chain_events, n_hard, n_total_compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_ROOT,
    )
    args = parser.parse_args()

    target_ids = (
        [int(s) for s in args.videos.split(",") if s.strip()]
        if args.videos else list(range(1, 20))
    )

    total_events = 0
    total_hard = 0
    total_compared = 0
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            print(f"[skip] v{vid:02d}: no match boundary")
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        ev, hard, compared = collect_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            out_dir=args.out_dir,
        )
        total_events += ev
        total_hard += hard
        total_compared += compared

    print(
        f"\n[summary] {total_events} chain events, "
        f"{total_hard}/{total_compared} hard cells "
        f"({100*total_hard/total_compared if total_compared else 0:.2f}%)"
    )
    print(f"[summary] saved to: {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
