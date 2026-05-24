"""CHAIN state 推論盤面 vs CNN raw 盤面の一致率を解析 (Phase B-14).

新方針 (project_recognition_strategy_pivot) では CHAIN 中は CNN を信用せず
ChainSimulator 推論を採用する設計。本スクリプトはその設計の妥当性を測る:

    - CHAIN state frame で InferenceBoardGenerator が返す推論盤面と
      CNN raw 盤面 (= ImageReader 出力) の cell 一致率を集計
    - 一致率が高い (= ChainSimulator が現実を再現) ならば推論主軸は機能
    - 一致率が低い (= シミュレーションと現実が乖離) なら、VideoChainTracker
      .before_board の精度に問題がある可能性高い

集計:
    - 動画別の CHAIN 中 cell 一致率
    - 連鎖段数別 (=何連鎖目か) の一致率分布
    - confusion matrix (推論色 → CNN 色)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_chain_inference_analysis \
        --duration 60 --fps 10 --per-video-model
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_UNKNOWN, HIDDEN_ROWS, VISIBLE_ROWS,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

_TARGET_ROWS = list(range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS))
_TARGET_COLS = list(range(BOARD_COLS))

_COLOR_NAME = {
    0: "EM", 1: "RD", 2: "BL", 3: "GR",
    4: "YE", 5: "PU", 9: "OJ", 10: "??",
}


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


def analyze_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int,
    cnn_model: Path | None,
) -> dict | None:
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return None
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    chain_frames = 0
    total_compared = 0
    total_match = 0
    confusion: Counter[tuple[int, int]] = Counter()  # (inferred, cnn)

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
            if side_res.state != BoardState.CHAIN:
                continue
            if side_res.inferred_board is None:
                continue
            chain_frames += 1
            inferred = side_res.inferred_board
            cnn = side_res.cnn_board
            for r in _TARGET_ROWS:
                for c in _TARGET_COLS:
                    inf_v = int(inferred.get(r, c))
                    cnn_v = int(cnn.get(r, c))
                    if inf_v == COLOR_UNKNOWN or cnn_v == COLOR_UNKNOWN:
                        continue
                    total_compared += 1
                    if inf_v == cnn_v:
                        total_match += 1
                    else:
                        confusion[(inf_v, cnn_v)] += 1
        frame_idx += 1
        t += interval
    cap.release()

    return {
        "video_id": video_id,
        "frames": frame_idx,
        "chain_frames": chain_frames,
        "total_compared": total_compared,
        "total_match": total_match,
        "match_rate": total_match / total_compared if total_compared else 0.0,
        "confusion": dict(confusion),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument("--cnn-model", type=Path, default=None)
    parser.add_argument("--per-video-model", action="store_true")
    parser.add_argument(
        "--out-tsv", type=Path,
        default=_ROOT / "data" / "phase_b_chain_inference.tsv",
    )
    args = parser.parse_args()

    target_ids = (
        [int(s) for s in args.videos.split(",") if s.strip()]
        if args.videos else list(range(1, 20))
    )

    rows: list[dict] = []
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        cnn_model = select_cnn_model(
            vid, args.per_video_model, args.cnn_model,
        )
        tag = f"CNN={cnn_model.name}" if cnn_model else "HSV"
        print(f"[run] v{vid:02d} ({tag}): [{start:.1f}, {end:.1f}]", flush=True)
        r = analyze_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            cnn_model=cnn_model,
        )
        if r is None:
            print(f"[fail] v{vid:02d}")
            continue
        if r["total_compared"] == 0:
            print(f"[empty] v{vid:02d}: no CHAIN frames")
        else:
            print(
                f"[done] v{vid:02d}: chain_frames={r['chain_frames']} "
                f"compared={r['total_compared']} "
                f"match_rate={r['match_rate']*100:.2f}%"
            )
        rows.append(r)

    if not rows:
        print("[empty] no data")
        return 0

    # 集計
    print()
    print("[summary] CHAIN state inferred vs cnn match rate")
    print(
        f"  {'video':<5}  {'chain_fr':>9}  {'compared':>9}  "
        f"{'match':>7}  {'rate':>7}"
    )
    total_compared = 0
    total_match = 0
    confusion_all: Counter[tuple[int, int]] = Counter()
    for r in rows:
        if r["total_compared"] == 0:
            print(
                f"  v{r['video_id']:02d}    {r['chain_frames']:>9d}  "
                f"{0:>9d}  {0:>7d}    n/a"
            )
            continue
        print(
            f"  v{r['video_id']:02d}    {r['chain_frames']:>9d}  "
            f"{r['total_compared']:>9d}  {r['total_match']:>7d}  "
            f"{r['match_rate']*100:>6.2f}%"
        )
        total_compared += r["total_compared"]
        total_match += r["total_match"]
        for k, v in r["confusion"].items():
            confusion_all[k] += v
    if total_compared:
        print(
            f"  {'AVG':<5}    {'':<9}  "
            f"{total_compared:>9d}  {total_match:>7d}  "
            f"{total_match/total_compared*100:>6.2f}%"
        )

    # confusion top
    print()
    print("[confusion] top 15 (inferred → cnn, count)")
    for (inf, cnn), n in sorted(
        confusion_all.items(), key=lambda kv: -kv[1],
    )[:15]:
        print(
            f"  {_COLOR_NAME.get(inf, '??'):>3} -> "
            f"{_COLOR_NAME.get(cnn, '??'):<3}: {n}"
        )

    # tsv 保存
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["video", "chain_frames", "compared", "match", "rate"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                "video": f"v{r['video_id']:02d}",
                "chain_frames": r["chain_frames"],
                "compared": r["total_compared"],
                "match": r["total_match"],
                "rate": f"{r['match_rate']*100:.3f}",
            })
    print(f"\n[saved] {to_windows_path(args.out_tsv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
