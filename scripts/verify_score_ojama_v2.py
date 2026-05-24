"""score OCR → 連鎖差分 → ojama 推論パイプラインの検証スクリプト。

video_01 (1080p) の指定試合について 0.5 秒間隔で score をサンプリングし、
infer_timeline_from_score_series で連鎖イベントと ojama 数を時系列推論する。

出力:
    data/verify/ojama_score_v2_video_01_match_<idx>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.ojama_score_inferrer import OjamaScoreInferrer
from src.score_ocr import ScoreOcr

DEFAULT_VIDEO = Path("data/frames/video_01.mp4")
DEFAULT_BOUNDARIES = Path("data/verify/match_boundaries_v4/video_01/matches.tsv")
DEFAULT_OUTDIR = Path("data/verify")


def sample_score_series(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    step_sec: float = 0.5,
) -> list[tuple[float, int, int]]:
    """指定区間で score を OCR で連続サンプリング。

    OCR 失敗 (None) の点は除外して返す。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画オープン失敗: {video_path}")
    ocr = ScoreOcr.load_default()
    series: list[tuple[float, int, int]] = []
    t = float(start_sec)
    while t <= float(end_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += step_sec
            continue
        res = ocr.read(frame)
        if res.score_1p is not None and res.score_2p is not None:
            series.append((t, int(res.score_1p), int(res.score_2p)))
        t += step_sec
    cap.release()
    return series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--boundaries", default=str(DEFAULT_BOUNDARIES))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--match-idx", type=int, default=None,
                        help="単一試合のみ処理する場合の idx")
    parser.add_argument("--max-matches", type=int, default=3,
                        help="match-idx 未指定時に処理する試合数 (先頭から)")
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    video = Path(args.video)
    bdy = Path(args.boundaries)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(bdy, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    if args.match_idx is not None:
        rows = [r for r in rows if int(r["idx"]) == args.match_idx]
    else:
        rows = rows[: args.max_matches]
    print(f"対象試合: {len(rows)} 件")

    for r in rows:
        idx = int(r["idx"])
        start = float(r["start_sec"])
        end = float(r["end_sec"])
        print(f"\n--- 試合 {idx}: {start:.1f}s - {end:.1f}s ---")
        series = sample_score_series(video, start, end, step_sec=args.step)
        print(f"  score 系列サンプル数: {len(series)} (OCR 成功のみ)")
        if not series:
            continue

        inferrer = OjamaScoreInferrer()
        preds = inferrer.infer_timeline_from_score_series(
            series, match_start_sec=start,
        )
        print(f"  推論連鎖イベント数: {len(preds)}")

        out = {
            "video": str(video),
            "match_idx": idx,
            "start_sec": start,
            "end_sec": end,
            "ocr_step_sec": args.step,
            "score_series_len": len(series),
            "predictions": [
                {
                    "fired_at_sec": p.fired_at_sec + start,
                    "elapsed_sec": p.fired_at_sec,
                    "fired_by": p.fired_by_side,
                    "side_receives": p.side,
                    "pending": p.pending,
                    "total_score": p.total_score,
                    "effective_rate": p.effective_rate,
                }
                for p in preds
            ],
            "leftover_1p": inferrer.leftover_1p,
            "leftover_2p": inferrer.leftover_2p,
        }
        out_path = outdir / f"ojama_score_v2_video_01_match_{idx:02d}.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  出力: {out_path}")
        # サマリ表示 (最初の 5 件)
        for p in preds[:5]:
            print(
                f"    t={p.fired_at_sec + start:6.1f}s {p.fired_by_side}発火 → "
                f"{p.side}に予告 {p.pending:3d}個 "
                f"(score+{p.total_score}, rate={p.effective_rate})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
