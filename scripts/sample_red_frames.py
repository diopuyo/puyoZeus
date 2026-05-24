"""
別試合動画から「赤ぷよが盤面にある」フレームだけをサンプルして
レビュー用画像一式を生成する。

動作:
    1. 動画を指定間隔でフレーム抽出（1920×1080 のみ採用）
    2. 各フレームで 1P / 2P を CNN 分類
    3. 赤判定が両サイド合計 `--min-red` 以上あるフレームだけ保存
    4. 保存フレームに対して verify_color_classification の画像生成を呼ぶ

使い方:
    ./venv/bin/python scripts/sample_red_frames.py \\
        --video data/frames/video_02.mp4 \\
        --interval 15 --max 6 --min-red 2

出力:
    data/frames/review_<video_stem>/frame_<mmss>s.png  ... 生フレーム
    data/verify/color_review_<video_stem>/...          ... 注釈付き画像
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    HIDDEN_ROWS,
)
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.match_state import MatchStateDetector
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier

DEFAULT_CNN: Path = Path("models/cnn_global_best.pt")
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")


def _count_red(
    frame: np.ndarray,
    region: BoardRegion,
    gated: GatedCnnClassifier,
) -> int:
    count = 0
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            patch = frame[y1c:y2c, x1c:x2c]
            if gated.classify(patch) == COLOR_RED:
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--interval", type=float, default=15.0, help="抽出間隔（秒）")
    parser.add_argument("--max", type=int, default=6, help="最大何フレーム保存するか")
    parser.add_argument("--min-red", type=int, default=2,
                        help="赤判定が両サイド合計でこの値以上のフレームのみ採用")
    parser.add_argument("--start", type=float, default=30.0, help="開始秒")
    parser.add_argument("--out-subdir", default=None, help="出力サブディレクトリ名")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"動画が存在しません: {video_path}", file=sys.stderr)
        return 1

    cnn = CnnPatchClassifier.load(DEFAULT_CNN)
    config = CalibratedConfig.load(DEFAULT_CALIB)
    gated = GatedCnnClassifier(color_classifier=cnn)
    match_det = MatchStateDetector(config.p1_region, config.p2_region)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"動画: {video_path.name}  duration={duration:.0f}s  fps={fps:.1f}")

    out_stem = args.out_subdir or f"review_{video_path.stem}"
    out_dir = Path("data/frames") / out_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    t = args.start
    while t < duration and len(saved) < args.max:
        # シークして 1 フレーム読む
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += args.interval
            continue
        # 1920×1080 にリサイズ（video_02 はすでに 1080p だが保険）
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        # 試合中フレームのみ採用（ブラウザ / VS 画面 / メニューを除外）
        mresult = match_det.detect(frame)
        if mresult.state.value != "in_match":
            t += args.interval
            continue

        n1 = _count_red(frame, config.p1_region, gated)
        n2 = _count_red(frame, config.p2_region, gated)
        total = n1 + n2
        if total >= args.min_red:
            tts = f"{int(t):04d}s"
            dest = out_dir / f"frame_{tts}.png"
            cv2.imwrite(str(dest), frame)
            saved.append(dest)
            print(f"  t={t:.1f}s  赤1P={n1} 赤2P={n2} → {dest.name}")
        t += args.interval

    cap.release()

    print(f"\n採用 {len(saved)} フレーム")
    if not saved:
        print("赤ぷよを含むフレームが見つかりませんでした")
        return 0

    # verify_color_classification を各フレームに対して呼ぶ
    # この script は複数パス対応しているのでまとめて渡す
    cmd = ["./venv/bin/python", "scripts/verify_color_classification.py"]
    cmd.extend(str(p) for p in saved)
    print(f"\n画像生成: {' '.join(cmd[:2])} [{len(saved)} frames]")
    subprocess.run(cmd, check=True)

    print("\n=== 生成済み color_review ===")
    for p in saved:
        review_path = Path("data/verify") / f"color_review_{p.stem}.png"
        print(f"  {review_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
