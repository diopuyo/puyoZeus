"""Phase I: 動画から HiddenRowValidator で擬似ラベルを収集する.

usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_hidden_row_collect \
        --video data/frames/video_02 \
        --video-id video_02 \
        --max-frames 5000

各動画の RecognitionPipeline を回しながら HiddenRowValidator にも update() を
投げ、検出された RevealEvent を LabelStore に永続化する。

出力:
    data/pseudo_labels/{video_id}/hidden_row.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from src.frame_reader import FrameReader  # noqa: E402
from src.recognition_pipeline import (  # noqa: E402
    RecognitionPipeline,
)
from src.self_supervised.hidden_row_validator import (  # noqa: E402
    HiddenRowValidator,
)
from src.self_supervised.label_store import LabelStore  # noqa: E402

# ============================
# 定数
# ============================

# デフォルトの 1 動画あたり最大処理 frame 数 (None で全 frame)
DEFAULT_MAX_FRAMES: int | None = None

# label store flush 間隔 (frame 数)
DEFAULT_FLUSH_INTERVAL: int = 500


def _iter_frames(
    video_path: Path, max_frames: int | None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """動画から (frame_idx, t_sec, frame_bgr) を順次 yield する."""
    reader = FrameReader(video_path)
    fps = reader.fps if reader.fps > 0 else 30.0
    for idx, frame in enumerate(reader):
        if max_frames is not None and idx >= max_frames:
            break
        yield idx, idx / fps, frame


def collect(
    video_path: Path,
    video_id: str,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    flush_interval: int = DEFAULT_FLUSH_INTERVAL,
) -> dict:
    """動画 1 本を処理し、hidden_row 擬似ラベルを収集 + 保存する.

    Returns:
        {n_frames, n_samples, video_id}
    """
    pipeline = RecognitionPipeline.load_default()
    validator = HiddenRowValidator()
    store = LabelStore(video_id=video_id)
    n_frames = 0
    n_samples = 0
    pending_buffer: list = []
    for frame_idx, t_sec, frame in _iter_frames(video_path, max_frames):
        result = pipeline.update(frame_idx, t_sec, frame)
        try:
            validator.update(frame_idx, t_sec, result, frame)
        except Exception:
            # silent skip: 擬似ラベル抽出失敗は本流に影響させない
            pass
        new_samples = validator.collect()
        if new_samples:
            pending_buffer.extend(new_samples)
            n_samples += len(new_samples)
        n_frames += 1
        if frame_idx % flush_interval == 0 and pending_buffer:
            store.append(pending_buffer)
            pending_buffer = []
    # 残りを flush
    if pending_buffer:
        store.append(pending_buffer)
    return {
        "video_id": video_id,
        "n_frames": n_frames,
        "n_samples": n_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase I: hidden_row 擬似ラベル収集",
    )
    parser.add_argument(
        "--video", type=Path, required=True,
        help="動画ファイルまたは frames ディレクトリ",
    )
    parser.add_argument(
        "--video-id", type=str, required=True,
        help="LabelStore 用の video_id",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="最大処理 frame 数 (省略で全 frame)",
    )
    parser.add_argument(
        "--flush-interval", type=int, default=DEFAULT_FLUSH_INTERVAL,
        help="LabelStore への flush 間隔 (frame 数)",
    )
    args = parser.parse_args(argv)
    result = collect(
        video_path=args.video,
        video_id=args.video_id,
        max_frames=args.max_frames,
        flush_interval=args.flush_interval,
    )
    print(f"[phase_i_collect] result={result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
