"""修正1(スコアOCRキャッシュ)+修正2(テロップ検出キャッシュ)の bit-identical 検証。

背景 (2026-07-30):
    src/recognition_pipeline.py + src/image_reader.py の重複排除修正が
    認識結果 (盤面グリッド・スコア値・state) を一切変えないことを、
    実動画で確認する。 本スクリプト自体は src/ を変更しない (読み取り専用)。

使い方 (WSL):
    修正前 (git stash 状態) と修正後の両方でこのスクリプトを実行し、
    出力される digest ファイルを diff する。

    nice -n 19 ./venv/bin/python -m scripts._diag_bitidentical_cache_2026-07-30 \
        --video data/frames/video_c60.mp4 --start-sec 1800 --seconds 30 \
        --out /tmp/digest_before.txt
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.recognition_pipeline import RecognitionPipeline, SideResult

TARGET_W: int = 1920
TARGET_H: int = 1080


def _board_digest(board: object) -> str:
    """Board を確定的な文字列に変換する (None なら 'None')。"""
    if board is None:
        return "None"
    rows = []
    for r in range(13):
        rows.append("".join(str(int(board.get(r, c))) for c in range(6)))
    return "|".join(rows)


def _side_digest(sr: "SideResult") -> str:
    """1 サイド分の SideResult を 1 行の digest 文字列にする。"""
    chain_tag = "None"
    if sr.chain_event is not None:
        ce = sr.chain_event
        chain_tag = (
            f"chain={ce.chain_count},score={ce.total_score},"
            f"ojama={ce.ojama_sent}"
        )
    return (
        f"state={sr.state.name} score={sr.score} delta={sr.score_delta} "
        f"cnn={_board_digest(sr.cnn_board)} "
        f"confirmed={_board_digest(sr.confirmed_board)} "
        f"chain=[{chain_tag}]"
    )


def build_pipeline() -> RecognitionPipeline:
    """本番相当の構成で pipeline を作る (render と同一設定)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
    )


def run_and_digest(
    video: str, start_sec: float, seconds: float,
) -> list[str]:
    """指定区間を処理し、frame ごとの digest 行リストを返す。"""
    pipeline = build_pipeline()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    n_frames = int(seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    lines: list[str] = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        fi = start_frame + i
        t = fi / fps
        result = pipeline.update(fi, t, frame)
        line = (
            f"f={fi} t={t:.3f} active={result.is_match_active} "
            f"1P[{_side_digest(result.p1)}] 2P[{_side_digest(result.p2)}]"
        )
        lines.append(line)
    cap.release()
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1800.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = run_and_digest(args.video, args.start_sec, args.seconds)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} frame 分の digest を {args.out} に書き出した。")


if __name__ == "__main__":
    main()
