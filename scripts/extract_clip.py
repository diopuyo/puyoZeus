"""
動画クリップ抽出ユーティリティ (ffmpeg 不要)

OpenCV のみで動画から時間範囲を切り出して MP4 ファイルに書き出す。
ffmpeg がない環境 (Windows 標準等) でも実行可能。

Usage:
    python scripts/extract_clip.py INPUT OUTPUT --start SEC --duration SEC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

# ============================
# 定数定義
# ============================

DEFAULT_FOURCC: str = "mp4v"
FALLBACK_FPS: float = 30.0
PROGRESS_REPORT_EVERY: int = 60  # フレームごとに進捗ログ


# ============================
# クリップ抽出
# ============================


def extract_clip(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    fourcc: str = DEFAULT_FOURCC,
) -> tuple[int, float]:
    """
    入力動画の [start, start+duration] を切り出して保存する。

    Args:
        input_path: 入力動画パス。
        output_path: 出力動画パス (MP4)。
        start_sec: 開始秒。
        duration_sec: クリップ長 (秒)。
        fourcc: VideoWriter 用 fourcc。

    Returns:
        (書き出しフレーム数, 出力 fps)。
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        end_sec = start_sec + duration_sec

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(
            str(output_path), fourcc_code, fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {output_path}")
        try:
            written = _write_range(
                cap=cap,
                writer=writer,
                start_sec=start_sec,
                end_sec=end_sec,
                fps=fps,
            )
        finally:
            writer.release()
    finally:
        cap.release()
    return written, fps


def _write_range(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> int:
    """指定秒範囲のフレームを writer に書き込む。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # 現在時刻 (秒)
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_sec = pos_ms / 1000.0 if pos_ms > 0 else (
            (start_sec + written / fps) if fps > 0 else start_sec
        )
        if pos_sec > end_sec:
            break
        writer.write(frame)
        written += 1
        if written % PROGRESS_REPORT_EVERY == 0:
            print(
                f"[extract_clip] {written} frames written "
                f"(t={pos_sec:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
    return written


# ============================
# CLI
# ============================


def _build_parser() -> argparse.ArgumentParser:
    """CLI パーサ。"""
    p = argparse.ArgumentParser(
        prog="extract_clip",
        description="OpenCV で動画から時間範囲を切り出す",
    )
    p.add_argument("input", help="入力動画パス")
    p.add_argument("output", help="出力動画パス (MP4)")
    p.add_argument(
        "--start", type=float, required=True, help="開始秒",
    )
    p.add_argument(
        "--duration", type=float, required=True, help="クリップ長 (秒)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI エントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"[ERROR] 入力動画なし: {in_path}", file=sys.stderr)
        return 1
    written, fps = extract_clip(
        in_path, out_path, args.start, args.duration,
    )
    print(
        f"[OK] {out_path} ({written} frames, fps={fps:.2f})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
