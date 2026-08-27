"""
得点ベース予告お邪魔ぷよ推論の検証スクリプト。

video_02 の matches.tsv から指定試合の連鎖イベントを VideoChainTracker で抽出し、
OjamaScoreInferrer で予告お邪魔の時系列を生成して JSON 出力する。

使い方:
    PYTHONPATH=. ./venv/bin/python scripts/verify_ojama_score_inference.py
    PYTHONPATH=. ./venv/bin/python scripts/verify_ojama_score_inference.py --idx 10
    PYTHONPATH=. ./venv/bin/python scripts/verify_ojama_score_inference.py --video data/frames/video_02.mp4 --idx 5

出力形式 (data/verify/ojama_score_video_02_match_<idx>.json):
    [
        {
            "fired_at_sec": float,        # 試合開始からの経過秒
            "fired_by": "1P" | "2P",
            "chain_length": int,
            "pending": int,                # 推論された予告お邪魔個数
            "target_side": "1P" | "2P",    # 受ける側 (= fired_by の反対)
            "total_score": int,
            "all_clear_bonus_applied": int,
            "is_all_clear": bool,
            "effective_rate": int
        },
        ...
    ]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import cv2

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chain_detector import VideoChainTracker
from src.image_reader import ImageReader
from src.ojama_score_inferrer import OjamaScoreInferrer
from src.sampling_config import BOARD_INTERVAL_SEC


DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "frames" / "video_02.mp4"
DEFAULT_BOUNDARIES_TSV = (
    PROJECT_ROOT / "data" / "verify" / "match_boundaries_v4" / "video_02"
    / "matches.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "verify"


def load_match_range(tsv_path: Path, idx: int) -> tuple[float, float]:
    """matches.tsv から指定 idx の (start_sec, end_sec) を返す。"""
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["idx"]) == idx:
                return float(row["start_sec"]), float(row["end_sec"])
    raise ValueError(f"matches.tsv に idx={idx} が見つかりません: {tsv_path}")


def collect_chain_events(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    interval_sec: float = BOARD_INTERVAL_SEC,
) -> list[tuple[float, str, "VideoChainTracker"]]:
    """指定区間で 1P/2P それぞれの連鎖イベントを抽出する。

    Returns:
        list of (fired_at_sec_within_match, side, ChainEvent)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    try:
        reader = ImageReader()
        # W38 注記 (2026-08-25 実測確認): 本スクリプトは tracker.update() に
        # 試合相対時刻 t (= abs_t - start_sec) を渡すため、match_start_sec=0.0
        # で正しい (elapsed = 試合相対)。pipeline 側の W38 (動画絶対時刻に
        # なる配線漏れ) の影響下には無い — video_97 試合5 (絶対417s〜) で
        # effective_rate=70 を実測確認済み (レート1張り付きは起きていない)。
        tracker_1p = VideoChainTracker(match_start_sec=0.0)
        tracker_2p = VideoChainTracker(match_start_sec=0.0)
        results: list[tuple[float, str, object]] = []

        t = 0.0
        while start_sec + t < end_sec:
            abs_t = start_sec + t
            cap.set(cv2.CAP_PROP_POS_MSEC, abs_t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += interval_sec
                continue
            try:
                board_1p, board_2p = reader.read_both_boards(frame)
            except Exception:
                t += interval_sec
                continue
            try:
                ev1 = tracker_1p.update(t, board_1p)
                if ev1 is not None:
                    results.append((t, "1P", ev1))
            except Exception:
                pass
            try:
                ev2 = tracker_2p.update(t, board_2p)
                if ev2 is not None:
                    results.append((t, "2P", ev2))
            except Exception:
                pass
            t += interval_sec
        return results
    finally:
        cap.release()


def build_predictions(
    chain_events: list[tuple[float, str, object]],
) -> list[dict]:
    """連鎖イベントから予告お邪魔の時系列を生成する。

    Note: ChainEvent から ChainResult を再構築するのは重いので、
    既に VideoChainTracker が得点・お邪魔送出を内部で計算した値
    (ev.total_score / ev.ojama_sent) を信頼してそのまま使う。
    OjamaScoreInferrer は ChainResult を要求するため、本スクリプトでは
    ev.before_board を再 simulate してから流す。
    """
    from src.chain import ChainSimulator

    sim = ChainSimulator()
    inferrer = OjamaScoreInferrer()

    # (fired_at, fired_by, ChainResult, all_clear_override) のリストへ整形
    timeline_input: list[tuple[float, str, object, bool]] = []
    for fired_at, side, ev in sorted(chain_events, key=lambda x: x[0]):
        cr = sim.simulate(ev.before_board)
        if cr.chain_count <= 0:
            continue
        timeline_input.append((fired_at, side, cr, False))

    preds = inferrer.infer_timeline(timeline_input, match_start_sec=0.0)

    out: list[dict] = []
    for pred in preds:
        out.append({
            "fired_at_sec": round(pred.fired_at_sec, 3),
            "fired_by": pred.fired_by_side,
            "chain_length": pred.chain_length,
            "pending": pred.pending,
            "target_side": pred.side,
            "total_score": pred.total_score,
            "all_clear_bonus_applied": pred.all_clear_bonus_applied,
            "is_all_clear": pred.is_all_clear,
            "effective_rate": pred.effective_rate,
        })
    return out


def summarize(predictions: list[dict]) -> dict:
    """簡易統計を返す (median pending、合計、件数)。"""
    pendings = [p["pending"] for p in predictions]
    by_side: dict[str, list[int]] = {"1P": [], "2P": []}
    for p in predictions:
        by_side[p["fired_by"]].append(p["pending"])
    return {
        "event_count": len(predictions),
        "total_pending_to_2p": sum(by_side["1P"]),
        "total_pending_to_1p": sum(by_side["2P"]),
        "median_pending_overall": (
            statistics.median(pendings) if pendings else 0
        ),
        "max_pending_overall": max(pendings) if pendings else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"対象動画パス (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=DEFAULT_BOUNDARIES_TSV,
        help=f"試合境界 TSV (default: {DEFAULT_BOUNDARIES_TSV})",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=10,
        help="試合インデックス (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 JSON パス (default: data/verify/ojama_score_video_02_match_<idx>.json)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"動画が存在しません: {args.video}", file=sys.stderr)
        return 1
    if not args.boundaries.exists():
        print(f"境界 TSV が存在しません: {args.boundaries}", file=sys.stderr)
        return 1

    start, end = load_match_range(args.boundaries, args.idx)
    print(f"[INFO] match {args.idx}: {start:.1f}s - {end:.1f}s ({end - start:.1f}s)")

    chain_events = collect_chain_events(args.video, start, end)
    print(f"[INFO] 検出連鎖イベント数: {len(chain_events)}")

    predictions = build_predictions(chain_events)

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"ojama_score_video_02_match_{args.idx:02d}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[OK] 出力: {output_path}")

    summary = summarize(predictions)
    print("[SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
