"""STABLE 中の drift cell を hard negative として採取 (Phase B-10 pivot).

新方針 (project_recognition_strategy_pivot) で BoardStateMachine が STABLE
確定中は「直近 STABLE 盤面 = 真値」とみなせる。その frame で CNN が異なる
cell = 単 frame 誤認識 → hard negative。

chain_truth よりこちらの方が信頼度が高い:
    - 真値 = 連続多数決で確定済の盤面 (= STABLE state ロジックの出力)
    - 物理推論や VideoChainTracker への依存なし
    - drift 解析で確認済の hard sample がそのまま訓練データになる

出力: data/training/drift_truth/{video}_drift.npz
    images: (N, 16, 16, 3) BGR uint8
    labels: (N,) int8 (真値 = 直近 STABLE 盤面の cell)
    cnn_pred: (N,) int8 (drift 発生時の CNN 出力)
    side: (N,) "1P" or "2P"
    row, col: (N,) int8
    frame_idx: (N,) int32

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_collect_drift_dataset \
        --duration 60 --fps 10 --per-video-model
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
    BOARD_COLS, COLOR_UNKNOWN, HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

CELL_PX: int = 16
DEFAULT_OUT_ROOT: Path = _ROOT / "data" / "training" / "drift_truth"


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


def select_cnn_model(
    video_id: int, per_video: bool, single_model: Path | None,
) -> Path | None:
    if per_video:
        from src.per_video_model_selector import select_phase_b_model
        m = select_phase_b_model(video_id)
        return Path(m) if m else None
    return single_model


def collect_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int,
    cnn_model: Path | None, out_dir: Path,
    min_consec_drift: int = 1,
) -> tuple[int, int]:
    """STABLE 中の drift cell を採取.

    Args:
        min_consec_drift: 連続 N frame 同じ (true_v, cnn_v) drift が続いたら
            CNN 確信的誤認識とみなす。1 frame だけのぶれは「真値が古い」
            起因の可能性で除外。
    """
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        print(f"[skip] v{video_id:02d}: video not found")
        return 0, 0

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[fail] v{video_id:02d}")
        return 0, 0

    images: list[np.ndarray] = []
    labels: list[int] = []
    cnn_preds: list[int] = []
    sides: list[str] = []
    rows_list: list[int] = []
    cols_list: list[int] = []
    frame_indices: list[int] = []
    consec_counts: list[int] = []

    # cell ごとの直近 drift 状態 (連続性チェック)
    # key: (side, row, col) -> (last_true_v, last_cnn_v, consec_count)
    consec_state: dict[
        tuple[str, int, int], tuple[int, int, int]
    ] = {}

    n_total_compared = 0
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

        for side_label, side_res in (
            ("1P", result.p1), ("2P", result.p2),
        ):
            if side_res.state != BoardState.STABLE:
                continue
            if side_res.confirmed_board is None:
                continue
            true_b = side_res.confirmed_board
            cnn_b = side_res.cnn_board
            region: BoardRegion = (
                DEFAULT_P1_REGION if side_label == "1P"
                else DEFAULT_P2_REGION
            )
            for r in range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS):
                for c in range(BOARD_COLS):
                    n_total_compared += 1
                    true_v = int(true_b.get(r, c))
                    cnn_v = int(cnn_b.get(r, c))
                    key = (side_label, r, c)
                    if (
                        true_v == COLOR_UNKNOWN
                        or cnn_v == COLOR_UNKNOWN
                        or true_v == cnn_v
                    ):
                        # drift なし: 連続カウンタリセット
                        if key in consec_state:
                            del consec_state[key]
                        continue

                    # 同じ drift パターンが続けば consec_count を増やす
                    prev = consec_state.get(key)
                    if prev is not None and prev[0] == true_v and prev[1] == cnn_v:
                        cnt = prev[2] + 1
                    else:
                        cnt = 1
                    consec_state[key] = (true_v, cnn_v, cnt)

                    if cnt < min_consec_drift:
                        continue  # 連続性未達、まだ採取しない

                    # 連続閾値到達、cell 採取
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
                    rows_list.append(r)
                    cols_list.append(c)
                    frame_indices.append(frame_idx)
                    consec_counts.append(cnt)
        frame_idx += 1
        t += interval
    cap.release()

    n_hard = len(images)
    if n_hard == 0:
        print(f"[empty] v{video_id:02d}: 0/{n_total_compared} drift cells")
        return n_total_compared, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v{video_id:02d}_drift.npz"
    np.savez_compressed(
        out_path,
        images=np.array(images, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int8),
        cnn_pred=np.array(cnn_preds, dtype=np.int8),
        side=np.array(sides),
        row=np.array(rows_list, dtype=np.int8),
        col=np.array(cols_list, dtype=np.int8),
        frame_idx=np.array(frame_indices, dtype=np.int32),
        consec_count=np.array(consec_counts, dtype=np.int16),
    )
    rate = 100.0 * n_hard / n_total_compared if n_total_compared else 0.0
    print(
        f"[done] v{video_id:02d}: {n_hard}/{n_total_compared} drift cells "
        f"({rate:.2f}%) -> {to_windows_path(out_path)}"
    )
    return n_total_compared, n_hard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--cnn-model", type=Path, default=None,
        help="一律 CNN model path",
    )
    parser.add_argument(
        "--per-video-model", action="store_true",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_ROOT,
    )
    parser.add_argument(
        "--min-consec-drift", type=int, default=3,
        help="連続 N frame 同じ (true_v,cnn_v) drift があれば採取 "
             "(default 3、CNN 確信的誤認識のみフィルタ)",
    )
    args = parser.parse_args()

    target_ids = (
        [int(s) for s in args.videos.split(",") if s.strip()]
        if args.videos else list(range(1, 20))
    )

    total_compared = 0
    total_hard = 0
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            print(f"[skip] v{vid:02d}: no match boundary")
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        cnn_model = select_cnn_model(
            vid, args.per_video_model, args.cnn_model,
        )
        compared, hard = collect_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            cnn_model=cnn_model, out_dir=args.out_dir,
            min_consec_drift=args.min_consec_drift,
        )
        total_compared += compared
        total_hard += hard

    rate = 100.0 * total_hard / total_compared if total_compared else 0.0
    print(
        f"\n[summary] {total_hard}/{total_compared} drift cells "
        f"({rate:.2f}%) across {len(target_ids)} videos"
    )
    print(f"[summary] saved to: {to_windows_path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
