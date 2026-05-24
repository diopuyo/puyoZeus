"""DB pre-inject の効果を cell-level で数値化 (Phase I.c 効果定量化)。

baseline (online_hsv 無効) と DB (pre-inject) を同 frame で並走、
1P/2P STABLE 中の confirmed_board の差分 cell 数を集計する。

使い方:
    python scripts/measure_db_impact.py --video xxx.mp4 --hsv-state xxx.json --cnn-model xxx.pt --max-frames 1500
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--hsv-state", type=Path, required=True)
    p.add_argument("--cnn-model", type=Path,
                    default=Path("models/cnn_phase_b_finetuned.pt"))
    p.add_argument("--max-frames", type=int, default=1500)
    return p.parse_args()


def _make_pipeline(cnn_model: Path, hsv_state: Path | None):
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=cnn_model, force_in_match=True,
    )
    if hsv_state is not None:
        with hsv_state.open() as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state["per_video_ranges"].items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if isinstance(hc, HybridClassifier) and ranges:
            hc._hsv.set_color_ranges_from_simple(ranges)
            if pipe._online_hsv is not None:
                pipe._online_hsv_injected = True
    else:
        # baseline: online_hsv 無効化
        pipe._online_hsv = None
    return pipe


def main() -> None:
    args = parse_args()
    pipe_b = _make_pipeline(args.cnn_model, hsv_state=None)
    pipe_db = _make_pipeline(args.cnn_model, hsv_state=args.hsv_state)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(n_total, int(args.max_frames))
    print(f"[measure] video={args.video} target={n_target}")
    diff_cells_total = 0
    same_cells_total = 0
    by_color_change: Counter[tuple[int, int]] = Counter()
    n_stable_frames = 0
    for fi in range(n_target):
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        rb = pipe_b.update(fi, fi / fps, fr)
        rd = pipe_db.update(fi, fi / fps, fr)
        for sb, sd in [(rb.p1, rd.p1), (rb.p2, rd.p2)]:
            if (sb.state != BoardState.STABLE
                    or sd.state != BoardState.STABLE):
                continue
            cb = sb.confirmed_board
            cd = sd.confirmed_board
            if cb is None or cd is None:
                continue
            n_stable_frames += 1
            for r in range(13):
                for c in range(6):
                    vb = int(cb.get(r, c))
                    vd = int(cd.get(r, c))
                    if vb == vd:
                        same_cells_total += 1
                    else:
                        diff_cells_total += 1
                        by_color_change[(vb, vd)] += 1
    cap.release()
    total = same_cells_total + diff_cells_total
    print(
        f"[measure] DONE n_stable_frames={n_stable_frames} "
        f"total_cells={total} same={same_cells_total} "
        f"diff={diff_cells_total} ({100*diff_cells_total/max(1,total):.2f}%)",
    )
    print(
        "[measure] top color changes (baseline → DB):",
        dict(by_color_change.most_common(10)),
    )


if __name__ == "__main__":
    main()
