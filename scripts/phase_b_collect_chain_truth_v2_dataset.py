"""連鎖完了「後」の安定 frame で真値を取って hard negative 集める (Phase B-10).

旧スクリプト (`phase_b_collect_chain_truth_dataset.py`) は VideoChainTracker
が drop 観測した瞬間の cnn_board と final_board を比較していたが、その時刻の
cnn_board は連鎖中の中間盤面なので 37% も不一致が出た (タイミングずれ)。

本スクリプトでは ChainEvent 受信後 chain_count × 0.5 秒経過の **連鎖完了後
安定 frame** で別途認識を行い、final_board (物理推論真値) と比較する。

出力: data/training/chain_truth_v2/{video}_hard.npz
    images: (N, 16, 16, 3) BGR uint8
    labels: (N,) int8 (真値 = ChainSimulator final_board の cell)
    cnn_pred: (N,) int8 (連鎖完了後 frame での CNN 出力)
    side, row, col, frame_idx: 各 (N,)
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
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion, ImageReader,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

CELL_PX: int = 16
DEFAULT_OUT_ROOT: Path = _ROOT / "data" / "training" / "chain_truth_v2"
CHAIN_COMPLETION_DELAY_PER_STEP: float = 0.5  # 連鎖アニメ 1 段あたりの猶予秒


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
    """1 動画から chain_truth_v2 hard negative を抽出.

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
    # 別 reader: 完了 frame の再認識用 (pipeline 内部の reader と独立)
    reader = ImageReader(use_match_state=False)
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
                continue  # 同じ event を pipeline が hold 中、初出のみ
            n_chain_events += 1
            chain_result = sim.simulate(ev.before_board)
            if chain_result.chain_count == 0:
                if side_label == "1P":
                    last_event_id_1p = ev_id
                else:
                    last_event_id_2p = ev_id
                continue
            true_board = chain_result.final_board

            # 連鎖完了後の target_time で frame を再取得
            target_t = (
                ev.end_sec
                + CHAIN_COMPLETION_DELAY_PER_STEP * chain_result.chain_count
            )
            if target_t >= end_sec:
                if side_label == "1P":
                    last_event_id_1p = ev_id
                else:
                    last_event_id_2p = ev_id
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, target_t * 1000.0)
            ok2, target_frame = cap.read()
            if not ok2 or target_frame is None:
                if side_label == "1P":
                    last_event_id_1p = ev_id
                else:
                    last_event_id_2p = ev_id
                continue
            if target_frame.shape[:2] != (1080, 1920):
                target_frame = cv2.resize(
                    target_frame, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            target_cnn_1p, target_cnn_2p = reader.read_both_boards(target_frame)
            target_cnn = (
                target_cnn_1p if side_label == "1P" else target_cnn_2p
            )
            region: BoardRegion = (
                DEFAULT_P1_REGION if side_label == "1P"
                else DEFAULT_P2_REGION
            )
            for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
                for c in range(BOARD_COLS):
                    n_total_compared += 1
                    true_v = int(true_board.get(r, c))
                    cnn_v = int(target_cnn.get(r, c))
                    if true_v == cnn_v:
                        continue
                    x1, y1, x2, y2 = region.cell_sample_rect(r, c)
                    patch = target_frame[y1:y2, x1:x2]
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

            # cap の読み込み位置を ev 観測時刻に戻す (= 元のループ位置)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
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
