"""大 ROI 走査間引き (enable_large_roi_throttle) の検証。

目的 3 点:
  1. **既定 OFF の bit-identical 確認**: 旧版 (コミット d7fc6c2) を別モジュールとして
     読み込み、同一フレーム列に対して確定盤面・各種フラグが完全一致することを実測する。
     本番ファイルの差し替えは裏走行ジョブが実行時にコードを読み直すため禁止
     (memory feedback_never_git_stash_multiagent_2026-07-30 と同じ事故の類型)。
  2. **ON 時の速度削減量の実測**: OFF/ON の認識時間を同一フレーム列で比較する。
  3. **ON 時の乖離量の実測**: 間引きにより結果が変わる箇所を件数で出す
     (bit-identical にならないのは仕様なので、影響の大きさを数値で示す)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_large_roi_throttle_2026-07-30 \
        --video data/baseline_videos_v3/v29m2_buf15s.mp4 --frames 300
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 旧版モジュールの配置先 (git show で書き出す)
BASELINE_MODULE_NAME = "_recognition_pipeline_baseline_2026_07_30"
# 比較対象とする SideResult の属性 (盤面 + 判定フラグ)
COMPARED_SCALAR_FIELDS: tuple[str, ...] = (
    "state",
    "is_stable",
    "score",
    "ojama_pending",
)


def _load_baseline_module(path: Path) -> Any:
    """旧版 recognition_pipeline を別名モジュールとして読み込む。"""
    spec = importlib.util.spec_from_file_location(BASELINE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"旧版モジュールを読み込めない: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASELINE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _read_frames(video: Path, frames: int, start_sec: float) -> list[np.ndarray]:
    """動画から連続フレームを読み出す (認識は 1920x1080 前提なのでリサイズ)。"""
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
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        out.append(frame)
    cap.release()
    return out


def _board_of(side_result: Any) -> np.ndarray | None:
    """SideResult から確定盤面 grid を取り出す。"""
    board = getattr(side_result, "confirmed_board", None)
    if board is None:
        return None
    grid = getattr(board, "grid", None)
    return np.asarray(grid) if grid is not None else None


def _signature(result: Any) -> tuple:
    """1 フレームの結果を比較可能な tuple に落とす。"""
    parts: list[Any] = []
    for side in ("side_1p", "side_2p"):
        sr = getattr(result, side, None)
        if sr is None:
            parts.append(None)
            continue
        grid = _board_of(sr)
        parts.append(None if grid is None else grid.tobytes())
        parts.extend(repr(getattr(sr, f, None)) for f in COMPARED_SCALAR_FIELDS)
    parts.append(repr(getattr(result, "is_match_active", None)))
    return tuple(parts)


def _run(pipeline_cls: Any, frames: list[np.ndarray], **kwargs: Any) -> tuple[list[tuple], float]:
    """パイプラインを 1 本走らせ、(フレーム毎シグネチャ, 所要秒) を返す。"""
    pipe = pipeline_cls.load_default(**kwargs)
    sigs: list[tuple] = []
    t0 = time.perf_counter()
    for idx, frame in enumerate(frames):
        res = pipe.update(idx, idx / 30.0, frame)
        sigs.append(_signature(res))
    return sigs, time.perf_counter() - t0


def _compare(label: str, base: list[tuple], test: list[tuple]) -> int:
    """2 本の結果を比較して差分フレーム数を報告する。"""
    n = min(len(base), len(test))
    diff = [i for i in range(n) if base[i] != test[i]]
    pct = 100.0 * len(diff) / n if n else 0.0
    print(f"  {label}: 差分 {len(diff)}/{n} フレーム ({pct:.2f}%)")
    if diff:
        print(f"    最初の差分フレーム: {diff[:10]}")
    return len(diff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start-sec", type=float, default=20.0)
    ap.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("/tmp/bitid/recognition_pipeline_old.py"),
        help="旧版 recognition_pipeline.py (git show で書き出したもの)",
    )
    args = ap.parse_args()

    cv2.setNumThreads(1)
    frames = _read_frames(args.video, args.frames, args.start_sec)
    print(f"読み出しフレーム数: {len(frames)}  ({args.video.name} t={args.start_sec}s)")

    from src.recognition_pipeline import RecognitionPipeline as NewPipeline

    print("\n[1] 新版 既定OFF")
    new_off, t_new_off = _run(NewPipeline, frames)
    print(f"  所要 {t_new_off:.2f}s = {len(frames) / t_new_off:.2f} fps")

    print("\n[2] 旧版 (d7fc6c2)")
    if args.baseline_file.exists():
        old_mod = _load_baseline_module(args.baseline_file)
        old_sigs, t_old = _run(old_mod.RecognitionPipeline, frames)
        print(f"  所要 {t_old:.2f}s = {len(frames) / t_old:.2f} fps")
        n_diff = _compare("既定OFF vs 旧版 (0 件なら bit-identical)", old_sigs, new_off)
        print(f"  → 判定: {'bit-identical 確認' if n_diff == 0 else '★不一致あり (要調査)'}")
    else:
        print(f"  skip (旧版ファイル不在: {args.baseline_file})")

    print("\n[3] 新版 間引きON")
    new_on, t_new_on = _run(NewPipeline, frames, enable_large_roi_throttle=True)
    print(f"  所要 {t_new_on:.2f}s = {len(frames) / t_new_on:.2f} fps")
    gain = 100.0 * (t_new_off - t_new_on) / t_new_off if t_new_off else 0.0
    print(f"  → 速度: {gain:+.1f}% ({t_new_off:.2f}s → {t_new_on:.2f}s)")
    _compare("既定OFF vs 間引きON (仕様上 0 にならない)", new_off, new_on)


if __name__ == "__main__":
    main()
