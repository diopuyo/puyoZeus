"""UI マスクのセル限定 (ui_mask_cells) 本番配線の採否検証 (2026-07-30)。

背景:
    `ui_mask_cells` を渡しているのは診断スクリプトだけで、`load_default` の既定は
    None。**本番 (収集・レンダ) では絞り込みが一切効いていなかった。**
    実測で ui_mask の matchTemplate は 27,054回/60フレーム = 認識全体の 40.9%、
    セル限定すると 714回 = 2.0%、1フレーム 226.3ms → 122.1ms (fps 4.42 → 8.19)。

    引き継ぎメモリの「4.4→8.07fps を出荷済み」は誤りで、8.07 は診断スクリプト内で
    フラグ ON にした値だった。本番はずっと 4.4fps 側で動いていた。

採否の決め手 (速度ではない):
    セル限定は「(1,2) 以外のセルで UI マスク判定をしない」挙動変更なので、
    **確定盤面が変わらないこと**が受け入れ条件。変わる場合は
    「消えた誤検出」なのか「見落とした×印」なのかを位置で切り分ける。

    user 伝授: ×印が出るのは「左から3列目の12段目」= grid (row=1, col=2) のみ。
    既知の副作用: 静止背景アートによる col=0 の誤検出 32件
    (`_diag_ui_mask_fire_positions_2026-07-30.py`) は**消えるのが正しい**。

単一動画の結果で採否を判断しない規律のため 3 動画で測る。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_ui_mask_cells_ab_2026-07-30 \
        --videos video_c56 video_c60 video_c65 --start-sec 1500 --frames 300
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
# 立ち上がり (キャッシュ構築・CNN ロード) を速度集計から除くフレーム数
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


def _grids(result: object) -> dict[str, np.ndarray | None]:
    """1P/2P の確定盤面 grid を取り出す。"""
    out: dict[str, np.ndarray | None] = {}
    for side, attr in (("1P", "side_1p"), ("2P", "side_2p")):
        sr = getattr(result, attr, None)
        board = getattr(sr, "confirmed_board", None) if sr is not None else None
        grid = getattr(board, "grid", None) if board is not None else None
        out[side] = None if grid is None else np.asarray(grid).copy()
    return out


def _run(frames: list[np.ndarray], restrict: bool) -> tuple[list[dict], float]:
    """パイプラインを 1 本走らせ、(フレーム毎の盤面, 定常中央 ms) を返す。"""
    from src.recognition_pipeline import RecognitionPipeline
    from src.ui_mask import UI_MASK_TARGET_CELLS

    pipe = RecognitionPipeline.load_default(
        ui_mask_cells=UI_MASK_TARGET_CELLS if restrict else None,
    )
    boards: list[dict] = []
    times: list[float] = []
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = pipe.update(idx, idx / 30.0, frame)
        times.append((time.perf_counter() - t0) * 1000.0)
        boards.append(_grids(res))
    arr = np.asarray(times)
    steady = arr[WARMUP_FRAMES:] if arr.size > WARMUP_FRAMES else arr
    return boards, float(np.median(steady))


def _compare(off: list[dict], on: list[dict]) -> tuple[int, int, Counter]:
    """盤面差分を数える。

    Returns:
        (差分フレーム数, 差分セル総数, 差分位置の Counter[(side, row, col)])。
    """
    n_frames_diff = 0
    n_cells_diff = 0
    positions: Counter = Counter()
    for f_off, f_on in zip(off, on):
        frame_has_diff = False
        for side in ("1P", "2P"):
            g_off, g_on = f_off[side], f_on[side]
            if g_off is None and g_on is None:
                continue
            if g_off is None or g_on is None or g_off.shape != g_on.shape:
                frame_has_diff = True
                continue
            diff = np.argwhere(g_off != g_on)
            if diff.size:
                frame_has_diff = True
                n_cells_diff += len(diff)
                for r, c in diff:
                    positions[(side, int(r), int(c))] += 1
        if frame_has_diff:
            n_frames_diff += 1
    return n_frames_diff, n_cells_diff, positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["video_c56", "video_c60", "video_c65"])
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument("--start-sec", type=float, default=1500.0)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    cv2.setNumThreads(1)
    from src.ui_mask import UI_MASK_TARGET_CELLS

    print(f"限定対象セル (grid row, col) = {sorted(UI_MASK_TARGET_CELLS)}")
    print(f"窓: t={args.start_sec}s から {args.frames} フレーム / cv_threads=1\n")

    totals = {"frames": 0, "diff_frames": 0, "diff_cells": 0}
    all_positions: Counter = Counter()
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        if not path.exists():
            print(f"[skip] 動画不在: {path}")
            continue
        frames = _read_frames(path, args.frames, args.start_sec)
        if not frames:
            print(f"[skip] フレームを読めない: {path}")
            continue
        boards_off, ms_off = _run(frames, restrict=False)
        boards_on, ms_on = _run(frames, restrict=True)
        n_fd, n_cd, pos = _compare(boards_off, boards_on)
        all_positions.update(pos)
        totals["frames"] += len(frames)
        totals["diff_frames"] += n_fd
        totals["diff_cells"] += n_cd
        gain = 100.0 * (ms_off - ms_on) / ms_off if ms_off else 0.0
        print(
            f"{name}: {len(frames)}フレーム  "
            f"速度 {ms_off:.1f}ms → {ms_on:.1f}ms ({gain:+.1f}%, "
            f"{1000 / ms_off:.2f} → {1000 / ms_on:.2f} fps)"
        )
        print(
            f"  盤面差分: {n_fd}/{len(frames)} フレーム "
            f"({100.0 * n_fd / len(frames):.2f}%)  差分セル {n_cd}個"
        )
        if pos:
            top = ", ".join(
                f"{s} r{r}c{c}×{n}" for (s, r, c), n in pos.most_common(6)
            )
            print(f"  差分位置: {top}")

    print(f"\n=== 合計 ({len(args.videos)}動画) ===")
    print(
        f"盤面差分 {totals['diff_frames']}/{totals['frames']} フレーム "
        f"({100.0 * totals['diff_frames'] / max(1, totals['frames']):.2f}%)  "
        f"差分セル {totals['diff_cells']}個"
    )
    if not all_positions:
        print("→ 差分ゼロ。セル限定は確定盤面を一切変えない = 純粋な高速化。")
        return
    # 差分位置が限定対象セル以外に集中していれば「消えた誤検出」の可能性が高い
    off_target = {
        k: v for k, v in all_positions.items() if (k[1], k[2]) not in UI_MASK_TARGET_CELLS
    }
    on_target = {
        k: v for k, v in all_positions.items() if (k[1], k[2]) in UI_MASK_TARGET_CELLS
    }
    print(
        f"→ 限定対象セル({sorted(UI_MASK_TARGET_CELLS)})での差分: "
        f"{sum(on_target.values())}件  ← ここが 0 でないと×印を見落としている疑い"
    )
    print(
        f"→ 対象外セルでの差分: {sum(off_target.values())}件  "
        "← 誤検出が消えた可能性 (要目視)"
    )
    for (s, r, c), n in sorted(all_positions.items(), key=lambda kv: -kv[1])[:12]:
        tag = "限定対象" if (r, c) in UI_MASK_TARGET_CELLS else "対象外"
        print(f"    {s} r{r}c{c}: {n}件 ({tag})")


if __name__ == "__main__":
    main()
