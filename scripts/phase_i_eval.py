"""Phase I: fine-tune 前後で認識精度を比較する評価スクリプト.

設計:
    1. 動画 N 本に対して RecognitionPipeline を実行
    2. 各 frame で score / next / chain の出力を記録
    3. fine-tune 前後の (score readable率, next 一致率, chain detect率) を比較

簡易実装:
    - score readable rate: ScoreOcr.read() で score_1p/2p が non-None の割合
    - next 一致率: NextValidator の placement_trace_match 件数 / 全 STABLE → STABLE 件数
    - chain detect 率: ChainValidator の score_match=True 件数 / 全 chain end 件数

Usage:
    python scripts/phase_i_eval.py \
        --videos data/frames/video_02.mp4 \
        --max-frames 6000 \
        [--cnn-model-before models/cnn_global_best.pt] \
        [--cnn-model-after models/cnn_pseudo_finetuned.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.sampling_config import BOARD_INTERVAL_SEC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=str,
                         help="カンマ区切り動画パス")
    parser.add_argument("--max-frames", type=int, default=6000)
    parser.add_argument("--cnn-model-before", type=Path,
                         default=Path("models/cnn_global_best.pt"))
    parser.add_argument("--cnn-model-after", type=Path, default=None,
                         help="None なら before のみ評価")
    parser.add_argument("--sampling-interval-sec", type=float,
                         default=BOARD_INTERVAL_SEC)
    return parser.parse_args()


def _eval_one(
    video_path: Path, cnn_path: Path | None, max_frames: int, step_sec: float,
) -> dict:
    """1 動画の認識精度メトリクスを集計."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "error": "cap open failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(step_sec * fps)))
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=cnn_path if cnn_path and cnn_path.is_file() else None,
        load_score_ocr=True,
        enable_chain_tracker=True,
        load_next_detector=True,
        enable_pseudo_label=True,  # 評価のため validator を回す
    )
    metrics = {
        "video": video_path.name,
        "n_frames": 0,
        "n_score_1p_read": 0,
        "n_score_2p_read": 0,
        "n_stable_1p": 0,
        "n_stable_2p": 0,
        "placement_match": 0,
        "placement_correct": 0,
        "chain_score_match": 0,
        "chain_total": 0,
    }
    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % step == 0:
                t_sec = frame_idx / fps
                res = pipe.update(processed, t_sec, frame)
                metrics["n_frames"] += 1
                if res.p1.score is not None:
                    metrics["n_score_1p_read"] += 1
                if res.p2.score is not None:
                    metrics["n_score_2p_read"] += 1
                from src.board_state_machine import BoardState
                if res.p1.state == BoardState.STABLE:
                    metrics["n_stable_1p"] += 1
                if res.p2.state == BoardState.STABLE:
                    metrics["n_stable_2p"] += 1
                processed += 1
                if max_frames > 0 and processed >= max_frames:
                    break
            frame_idx += 1
        # validator から最終 collect
        samples = pipe.collect_pseudo_labels()
        for s in samples:
            src = s.metadata.get("source", "")
            if src == "placement_trace_match":
                metrics["placement_match"] += 1
            elif src == "placement_trace_correct":
                metrics["placement_correct"] += 1
            elif src == "chain_consistency":
                metrics["chain_total"] += 1
                if s.metadata.get("score_match"):
                    metrics["chain_score_match"] += 1
    finally:
        cap.release()
    return metrics


def _format(metrics: list[dict], label: str) -> str:
    lines = [f"=== {label} ==="]
    for m in metrics:
        if "error" in m:
            lines.append(f"{m['video']}: ERROR ({m['error']})")
            continue
        n = m["n_frames"] or 1
        lines.append(
            f"{m['video']}: frames={m['n_frames']} "
            f"score1p={m['n_score_1p_read']/n:.2%} "
            f"score2p={m['n_score_2p_read']/n:.2%} "
            f"placement_match={m['placement_match']} "
            f"placement_correct={m['placement_correct']} "
            f"chain_match={m['chain_score_match']}/{m['chain_total']}",
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    video_paths = [Path(p.strip()) for p in args.videos.split(",") if p.strip()]
    print(f"[phase_i] eval videos: {[p.name for p in video_paths]}")
    print(f"[phase_i] before model: {args.cnn_model_before}")
    metrics_before = [
        _eval_one(p, args.cnn_model_before, args.max_frames,
                   args.sampling_interval_sec)
        for p in video_paths
    ]
    print(_format(metrics_before, "BEFORE"))
    if args.cnn_model_after is not None:
        print(f"[phase_i] after model: {args.cnn_model_after}")
        metrics_after = [
            _eval_one(p, args.cnn_model_after, args.max_frames,
                       args.sampling_interval_sec)
            for p in video_paths
        ]
        print(_format(metrics_after, "AFTER"))


if __name__ == "__main__":
    main()
