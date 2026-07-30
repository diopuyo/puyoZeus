"""背景照合の直接ピアソン化 (ENABLE_DIRECT_PEARSON_NCC) の採否検証 (2026-07-31)。

直接ピアソンは演算順序が違うため **bit-identical ではない** (単体実測の差は 4.16e-17)。
背景照合は閾値 PATCH_NCC_EMPTY_THRESHOLD=0.92 で「空か否か」の二値判定に使われるので、
境界ぴったりでは判定が変わりうる。実動画で以下を測る:

  1. **読み取ったスコア値そのものの差分** (最重要。OCR が変われば
     おじゃま会計・連鎖検出・試合境界の全てに波及する)
  2. 確定盤面の差分
  3. 速度

単一動画で採否を判断しない規律のため 3 動画で測る。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_score_ocr_matmul_ab_2026-07-30 \
        --frames 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
WARMUP_FRAMES: int = 10


def _read_frames(video: Path, frames: int, start_sec: float) -> list[np.ndarray]:
    """動画から連続フレームを読み出す (1920x1080 に正規化)。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    out: list[np.ndarray] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        out.append(frame)
    cap.release()
    return out


def _snapshot(result: object) -> tuple:
    """1 フレームの「OCR スコア + 確定盤面」を比較可能な形にする。"""
    parts: list = []
    for attr in ("side_1p", "side_2p"):
        sr = getattr(result, attr, None)
        parts.append(getattr(sr, "score", None) if sr is not None else None)
        board = getattr(sr, "confirmed_board", None) if sr is not None else None
        grid = getattr(board, "grid", None) if board is not None else None
        parts.append(None if grid is None else np.asarray(grid).tobytes())
    return tuple(parts)


def _run(frames: list[np.ndarray], direct: bool) -> tuple[list[tuple], float]:
    """パイプラインを 1 本走らせ、(フレーム毎スナップショット, 定常中央 ms)。

    ENABLE_DIRECT_PEARSON_NCC はモジュール定数なので、走行前に切り替える
    (呼び出し側で必ず元に戻す)。
    """
    import src.background_fingerprint as bgfp
    from src.recognition_pipeline import RecognitionPipeline

    bgfp.ENABLE_DIRECT_PEARSON_NCC = direct
    pipe = RecognitionPipeline.load_default()
    snaps: list[tuple] = []
    times: list[float] = []
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = pipe.update(idx, idx / 30.0, frame)
        times.append((time.perf_counter() - t0) * 1000.0)
        snaps.append(_snapshot(res))
    arr = np.asarray(times)
    steady = arr[WARMUP_FRAMES:] if arr.size > WARMUP_FRAMES else arr
    return snaps, float(np.median(steady))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos", nargs="+", default=["video_c56", "video_c60", "video_c65"],
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--start-sec", type=float, default=1500.0)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    print(f"窓: t={args.start_sec}s から {args.frames} フレーム / cv_threads=1\n")

    tot_frames = 0
    tot_score_diff = 0
    tot_board_diff = 0
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        if not path.exists():
            print(f"[skip] 動画不在: {path}")
            continue
        frames = _read_frames(path, args.frames, args.start_sec)
        if not frames:
            print(f"[skip] フレームを読めない: {path}")
            continue
        off, ms_off = _run(frames, direct=False)
        on, ms_on = _run(frames, direct=True)

        n = min(len(off), len(on))
        score_diff = 0
        board_diff = 0
        examples: list[str] = []
        for i in range(n):
            # index 0,2 = score / 1,3 = board
            if off[i][0] != on[i][0] or off[i][2] != on[i][2]:
                score_diff += 1
                if len(examples) < 3:
                    examples.append(
                        f"f{i}: 1P {off[i][0]}→{on[i][0]} / 2P {off[i][2]}→{on[i][2]}"
                    )
            if off[i][1] != on[i][1] or off[i][3] != on[i][3]:
                board_diff += 1
        tot_frames += n
        tot_score_diff += score_diff
        tot_board_diff += board_diff
        gain = 100.0 * (ms_off - ms_on) / ms_off if ms_off else 0.0
        print(
            f"{name}: {n}フレーム  速度 {ms_off:.1f}ms → {ms_on:.1f}ms "
            f"({gain:+.1f}%, {1000 / ms_off:.2f} → {1000 / ms_on:.2f} fps)"
        )
        print(
            f"  スコア差分: {score_diff}/{n} フレーム  "
            f"盤面差分: {board_diff}/{n} フレーム"
        )
        for ex in examples:
            print(f"    {ex}")

    print(f"\n=== 合計 ({len(args.videos)}動画) ===")
    print(
        f"スコア差分 {tot_score_diff}/{tot_frames} フレーム "
        f"({100.0 * tot_score_diff / max(1, tot_frames):.2f}%)"
    )
    print(
        f"盤面差分 {tot_board_diff}/{tot_frames} フレーム "
        f"({100.0 * tot_board_diff / max(1, tot_frames):.2f}%)"
    )
    if tot_score_diff == 0 and tot_board_diff == 0:
        print("→ 差分ゼロ。演算順序の違い (4e-17) は空判定に影響していない。")
    else:
        print(
            "→ 差分あり。閾値境界で判定が変わった可能性。"
            "**既定ONにする前に差分フレームを目視すること。**"
        )


if __name__ == "__main__":
    main()
