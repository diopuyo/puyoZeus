"""Phase B-8: 全動画 (v01〜v19) で pipeline eval を回し統計を集約.

各動画の試合 1 区間の最初 N 秒で BoardStateMachine pipeline を回し、
state 分布 + drift 統計を tsv に保存する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_pipeline_eval_all \
        --duration 30 --fps 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time as time_mod
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.phase_b_pipeline_eval import SideStats  # noqa: E402

VIDEOS: list[int] = list(range(1, 20))

# CLI で --cnn-model を指定するとここに反映される (グローバル簡易実装)
_CNN_MODEL_PATH: Path | None = None
_PER_VIDEO_MODEL: bool = False
_TEMPORAL_SMOOTHING: int = 1


def get_match1(video_id: int) -> tuple[float, float] | None:
    """match_boundaries から試合 1 の (start, end) を返す."""
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
            row = rows[1]
            try:
                return float(row[1]), float(row[2])
            except (IndexError, ValueError):
                continue
    return None


def run_one(
    video_id: int, start_sec: float, end_sec: float, fps_sample: float,
    stable_n: int,
) -> dict[str, object] | None:
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return None

    # Per-video model: 動画別に HSV/CNN を切り替え (--per-video-model 指定時)
    if _PER_VIDEO_MODEL:
        from src.per_video_model_selector import (
            select_phase_b_model, select_phase_b_smoothing,
        )
        m = select_phase_b_model(video_id)
        cnn_model = Path(m) if m else None
        # per-video smoothing も適用 (改善動画のみ smoothing=3、他は 1)
        smoothing_n = select_phase_b_smoothing(video_id)
    else:
        cnn_model = _CNN_MODEL_PATH if _CNN_MODEL_PATH else None
        smoothing_n = _TEMPORAL_SMOOTHING

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=smoothing_n,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    interval = 1.0 / fps_sample
    p1_stats = SideStats(side="1P")
    p2_stats = SideStats(side="2P")
    total = 0
    t = start_sec
    frame_idx = 0
    t0 = time_mod.time()
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
        p1_stats.update(result.p1)
        p2_stats.update(result.p2)
        total += 1
        frame_idx += 1
        t += interval
    cap.release()
    elapsed = time_mod.time() - t0

    return {
        "video": f"v{video_id:02d}",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "frames": total,
        "elapsed_sec": elapsed,
        "p1": p1_stats,
        "p2": p2_stats,
    }


def stats_row(side_label: str, stats: SideStats, total: int) -> dict:
    def ratio(state: BoardState) -> float:
        return 100.0 * stats.state_counts.get(state, 0) / total if total else 0.0
    return {
        f"{side_label}_menu_pct": f"{ratio(BoardState.MENU):.1f}",
        f"{side_label}_stable_pct": f"{ratio(BoardState.STABLE):.1f}",
        f"{side_label}_tsumo_pct": f"{ratio(BoardState.TSUMO_FALL):.1f}",
        f"{side_label}_chain_pct": f"{ratio(BoardState.CHAIN):.1f}",
        f"{side_label}_ojama_pct": f"{ratio(BoardState.OJAMA_FALL):.1f}",
        f"{side_label}_effect_pct": f"{ratio(BoardState.EFFECT):.1f}",
        f"{side_label}_drift": stats.drift_frames,
        f"{side_label}_resync": stats.resync_frames,
        f"{side_label}_avg_mm": (
            f"{stats.total_mismatch / total:.2f}" if total else "0"
        ),
        f"{side_label}_chain_evt": stats.chain_event_count,
        f"{side_label}_uniq_conf": len(stats.unique_confirmed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument(
        "--videos", type=str, default="",
        help="カンマ区切り (例: 1,2,5)。空なら全 v01-v19",
    )
    parser.add_argument(
        "--out-tsv", type=Path,
        default=_ROOT / "data" / "phase_b_eval_summary.tsv",
    )
    parser.add_argument(
        "--cnn-model", type=Path, default=None,
        help="HybridClassifier 用 CNN model path (指定なしなら HSV のみ)",
    )
    parser.add_argument(
        "--per-video-model", action="store_true",
        help="動画 ID ごとに per_video_model_selector で HSV/CNN を切替",
    )
    parser.add_argument(
        "--temporal-smoothing", type=int, default=1,
        help="CNN 時系列平均 N frame (1=無効)",
    )
    args = parser.parse_args()
    global _CNN_MODEL_PATH, _PER_VIDEO_MODEL, _TEMPORAL_SMOOTHING
    _CNN_MODEL_PATH = args.cnn_model
    _PER_VIDEO_MODEL = args.per_video_model
    _TEMPORAL_SMOOTHING = args.temporal_smoothing

    if args.videos:
        target_ids = [int(s) for s in args.videos.split(",") if s.strip()]
    else:
        target_ids = VIDEOS

    rows: list[dict] = []
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            print(f"[skip] v{vid:02d}: no match boundary")
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        print(f"[run ] v{vid:02d}: [{start:.1f}, {end:.1f}] sec ", flush=True)
        res = run_one(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
        )
        if res is None:
            print(f"[fail] v{vid:02d}")
            continue
        total = int(res["frames"])
        elapsed = float(res["elapsed_sec"])
        print(
            f"[done] v{vid:02d}: {total} frames in {elapsed:.1f}s "
            f"({total/elapsed if elapsed > 0 else 0:.1f} fps)"
        )
        row = {
            "video": res["video"],
            "frames": total,
            "elapsed_sec": f"{elapsed:.1f}",
        }
        row.update(stats_row("1P", res["p1"], total))  # type: ignore[arg-type]
        row.update(stats_row("2P", res["p2"], total))  # type: ignore[arg-type]
        rows.append(row)

    if not rows:
        print("[summary] no rows")
        return 0

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {to_windows_path(args.out_tsv)}")

    # コンソール用 sticky summary (1P STABLE %)
    print()
    print("=== 1P STABLE distribution ===")
    for r in rows:
        print(
            f"  {r['video']}  STABLE={r['1P_stable_pct']:>5}%  "
            f"CHAIN={r['1P_chain_pct']:>5}%  "
            f"TSUMO={r['1P_tsumo_pct']:>5}%  "
            f"MENU={r['1P_menu_pct']:>5}%  "
            f"drift={r['1P_drift']}  resync={r['1P_resync']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
