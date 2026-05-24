"""全試合の score 時系列を OCR でサンプリングし JSON にキャッシュする。

出力:
    data/training/score_series_cache.json

形式:
    {
        "video_01": {
            "0": [{"t": 187.0, "1p": 0, "2p": 0}, ...],
            "1": [...],
            ...
        },
        "video_02": ...,
        "video_03": ...
    }

このキャッシュは後段の incoming_ojama 集計ステップで使う。
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

from src.score_ocr import ScoreOcr

VIDEO_IDS = ("video_01", "video_02", "video_03")
DEFAULT_VIDEO_DIR = Path("data/frames")
DEFAULT_BOUNDARY_DIR = Path("data/verify/match_boundaries_v4")
DEFAULT_OUT = Path("data/training/score_series_cache.json")
DEFAULT_STEP_SEC = 0.5


def sample_match(
    cap: cv2.VideoCapture,
    ocr: ScoreOcr,
    start_sec: float,
    end_sec: float,
    step_sec: float,
) -> list[dict]:
    """1 試合分の score 時系列を OCR でサンプリング。"""
    samples: list[dict] = []
    t = float(start_sec)
    while t <= float(end_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            res = ocr.read(frame)
            samples.append({
                "t": round(t, 2),
                "1p": res.score_1p,
                "2p": res.score_2p,
                "c1": round(res.confidence_1p, 3),
                "c2": round(res.confidence_2p, 3),
            })
        t += step_sec
    return samples


def process_video(
    video_id: str,
    video_path: Path,
    boundary_path: Path,
    ocr: ScoreOcr,
    step_sec: float,
) -> dict[str, list[dict]]:
    if not video_path.is_file():
        print(f"  [warn] 動画なし: {video_path}")
        return {}
    if not boundary_path.is_file():
        print(f"  [warn] matches.tsv なし: {boundary_path}")
        return {}
    rows: list[dict] = []
    with open(boundary_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    print(f"  [{video_id}] 試合数: {len(rows)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] 動画オープン失敗: {video_path}")
        return {}

    out: dict[str, list[dict]] = {}
    for r in rows:
        idx = r["idx"]
        start = float(r["start_sec"])
        end = float(r["end_sec"])
        samples = sample_match(cap, ocr, start, end, step_sec)
        ok_count = sum(1 for s in samples if s["1p"] is not None and s["2p"] is not None)
        out[idx] = samples
        print(f"  [{video_id}] match {idx}: "
              f"{len(samples)} samples ({ok_count} OCR OK)")
    cap.release()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--boundary-dir", default=str(DEFAULT_BOUNDARY_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_SEC)
    parser.add_argument("--video-ids", nargs="+", default=list(VIDEO_IDS))
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    bdy_dir = Path(args.boundary_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ocr = ScoreOcr.load_default()

    cache: dict[str, dict] = {}
    for vid in args.video_ids:
        print(f"\n=== {vid} ===")
        cache[vid] = process_video(
            video_id=vid,
            video_path=video_dir / f"{vid}.mp4",
            boundary_path=bdy_dir / vid / "matches.tsv",
            ocr=ocr,
            step_sec=args.step,
        )
    out_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_matches = sum(len(v) for v in cache.values())
    total_samples = sum(len(s) for v in cache.values() for s in v.values())
    print(f"\n出力: {out_path}")
    print(f"試合: {total_matches}, サンプル: {total_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
