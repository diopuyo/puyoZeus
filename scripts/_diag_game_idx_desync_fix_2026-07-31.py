"""game_idx desync 根治の効果を実動画で測る (2026-07-31)。

## 何を測るか

旧実装 (side 独立カウンタ) と新実装 (1P/2P 共有カウンタ) で、
**同じ game_idx が指す時刻がどれだけずれるか**を比較する。

指標:
  1. **game_idx ごとの 1P/2P 開始時刻の差** — これが desync の実体。
     2026-07-29 の実測では 57.6% のゲームが 5 秒超ずれていた。
  2. 検出ゲーム数 (1P/2P で一致するか)
  3. 片側が game 0 に留まる現象の有無 (score OCR 破綻動画)

## 使い方

    PYTHONPATH=. ./venv/bin/python -m scripts._diag_game_idx_desync_fix_2026-07-31 \
        --videos video_c60 video_c26 video_c58 --max-sec 600
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# 「大きくずれている」と判定する閾値 [秒] (2026-07-29 の集計と同じ基準)
DESYNC_THRESHOLD_SEC: float = 5.0


def _import_lean() -> Any:
    """collect_boards_lean をモジュールとして読み込む。"""
    path = Path(__file__).resolve().parent / "collect_boards_lean.py"
    spec = importlib.util.spec_from_file_location("_lean_for_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("collect_boards_lean を読み込めない")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lean_for_diag"] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect(
    mod: Any, video: Path, start_sec: float, max_sec: float, shared: bool,
) -> Any:
    """1 動画を収集して accumulator を返す (npz は書かない)。

    共有カウンタの有無を切り替えるため、`_process_side_lean` を薄くラップする。
    """
    import cv2

    from src.recognition_pipeline import RecognitionPipeline

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or mod.DEFAULT_FPS
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_frames = int(max_sec * fps)
    acc = mod._LeanNpzAccumulator()
    s1, s2 = mod._SideState(), mod._SideState()
    shared_counter = mod._SharedGameCounter() if shared else None
    pipe = RecognitionPipeline.load_default()
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (mod.TARGET_H, mod.TARGET_W):
            frame = cv2.resize(
                frame, (mod.TARGET_W, mod.TARGET_H), interpolation=cv2.INTER_AREA,
            )
        fi = start_frame + i
        t_sec = fi / fps
        res = pipe.update(fi, t_sec, frame)
        for state, label, side_res in (
            (s1, "1P", res.p1), (s2, "2P", res.p2),
        ):
            mod._process_side_lean(
                acc, state, label, side_res.confirmed_board, side_res.state,
                side_res.score, video.stem, t_sec, fi,
                shared_game=shared_counter,
            )
    cap.release()
    return acc


def _desync_stats(acc: Any) -> dict[str, Any]:
    """game_idx ごとの 1P/2P 開始時刻差を集計する。"""
    first_t: dict[tuple[int, str], float] = {}
    for gidx, side, t in zip(acc.game_idxs, acc.sides, acc.t_secs):
        key = (int(gidx), str(side))
        if key not in first_t or t < first_t[key]:
            first_t[key] = float(t)
    gaps: list[float] = []
    per_side_games: dict[str, set[int]] = defaultdict(set)
    for (gidx, side) in first_t:
        per_side_games[side].add(gidx)
    common = per_side_games["1P"] & per_side_games["2P"]
    for gidx in sorted(common):
        gaps.append(abs(first_t[(gidx, "1P")] - first_t[(gidx, "2P")]))
    arr = np.asarray(gaps) if gaps else np.array([])
    return {
        "n_rows": len(acc.game_idxs),
        "games_1p": len(per_side_games["1P"]),
        "games_2p": len(per_side_games["2P"]),
        "games_common": len(common),
        "gaps": arr,
    }


def _fmt(stats: dict[str, Any]) -> str:
    """集計結果を 1 行にまとめる。"""
    arr = stats["gaps"]
    if arr.size == 0:
        return (
            f"行数 {stats['n_rows']:>5}  "
            f"ゲーム数 1P={stats['games_1p']} 2P={stats['games_2p']} "
            f"共通={stats['games_common']}  (共通ゲームなし)"
        )
    over = int((arr > DESYNC_THRESHOLD_SEC).sum())
    return (
        f"行数 {stats['n_rows']:>5}  "
        f"ゲーム数 1P={stats['games_1p']} 2P={stats['games_2p']} "
        f"共通={stats['games_common']}  "
        f"時刻差 中央 {float(np.median(arr)):.2f}s 最大 {float(arr.max()):.2f}s  "
        f"{DESYNC_THRESHOLD_SEC:.0f}s超 {over}/{arr.size} "
        f"({100.0 * over / arr.size:.1f}%)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos", nargs="+",
        default=["video_c60", "video_c26", "video_c58"],
        help="既定は通常動画 + score OCR 破綻の既知難物 c26/c58",
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--max-sec", type=float, default=600.0)
    args = ap.parse_args()

    import cv2

    cv2.setNumThreads(1)
    mod = _import_lean()
    print(
        f"窓: t={args.start_sec}s から {args.max_sec}s / "
        f"ずれ判定閾値 {DESYNC_THRESHOLD_SEC}s\n"
    )
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        if not path.exists():
            print(f"[skip] 動画不在: {name}")
            continue
        for label, shared in (("旧(side独立)", False), ("新(共有)", True)):
            acc = _collect(
                mod, path, args.start_sec, args.max_sec, shared=shared,
            )
            print(f"{name:<12} {label:<14} {_fmt(_desync_stats(acc))}")
        print()


if __name__ == "__main__":
    main()
