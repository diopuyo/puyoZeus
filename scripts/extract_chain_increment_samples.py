"""連鎖中フレーム + 全消し時フレームのサンプルを取得して、ROI 確認用 grid を作る。

目的:
    - "+N" 連鎖獲得点数表示の位置を確認
    - 「全消し！」演出表示の位置を確認

戦略:
    - score_series_cache から大連鎖イベントを抽出
    - 連鎖発火時刻の +0.0s, +0.5s, +1.0s, +1.5s, +2.0s を 5 frame サンプル
    - 1P/2P 両側、計 5×2=10 frames per match
    - 6 試合分を grid に並べる
"""
from __future__ import annotations

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
import numpy as np

CACHE_PATH = Path("data/training/score_series_cache.json")
OUT_DIR = Path("data/verify/chain_increment_samples")
OUT_GRID = Path("data/verify/chain_increment_grid.png")
VIDEO_DIR = Path("data/frames")
N_MATCHES = 6
N_FRAMES_PER = 5  # 連鎖発火時刻からの相対秒
TIME_OFFSETS = (0.0, 0.5, 1.0, 1.5, 2.0)


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def find_large_chains(cache: dict, min_delta: int = 5000) -> list[tuple]:
    events: list[tuple] = []
    for vid, matches in cache.items():
        for midx, samples in matches.items():
            valid = [s for s in samples
                     if s["1p"] is not None and s["2p"] is not None]
            for i in range(1, len(valid)):
                d1 = valid[i]["1p"] - valid[i - 1]["1p"]
                d2 = valid[i]["2p"] - valid[i - 1]["2p"]
                if d1 >= min_delta:
                    events.append((vid, midx, valid[i]["t"], "1P", d1))
                if d2 >= min_delta:
                    events.append((vid, midx, valid[i]["t"], "2P", d2))
    events.sort(key=lambda x: x[4], reverse=True)
    return events


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    events = find_large_chains(cache)
    print(f"大連鎖イベント: {len(events)}")

    # 多様性のため異なる試合から N_MATCHES 採用
    selected: list = []
    seen: set = set()
    for ev in events:
        key = (ev[0], ev[1])
        if key in seen:
            continue
        seen.add(key)
        selected.append(ev)
        if len(selected) >= N_MATCHES:
            break

    panels: list[np.ndarray] = []
    for fi, (vid, midx, t_sec, side, delta) in enumerate(selected):
        # 連鎖発火直前 -1s から +2s まで 5 サンプル (発火 -0.5, 0, +0.5, +1, +2)
        offsets_ext = (-0.5, 0.0, 0.5, 1.0, 2.0)
        small_panels: list[np.ndarray] = []
        for off in offsets_ext:
            t = t_sec + off
            frame = get_frame(VIDEO_DIR / f"{vid}.mp4", t)
            if frame is None:
                continue
            # フル画面を 0.4x にリサイズ (768x432)
            small = cv2.resize(frame, (768, 432), interpolation=cv2.INTER_AREA)
            # ヘッダ
            bar = np.zeros((26, small.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                bar, f"F{fi} {vid} m{midx} t={t:.1f}s "
                f"({side}+{off:+.1f}s, delta={delta})",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 200, 100), 1,
            )
            small_panels.append(np.vstack([bar, small]))
            # 個別フレームも保存
            cv2.imwrite(
                str(OUT_DIR / f"F{fi}_{vid}_m{midx}_t{t:.1f}s.png"),
                frame,
            )
        # 横結合 (5 frames)
        sep = np.full((small_panels[0].shape[0], 4, 3), 60, dtype=np.uint8)
        parts: list[np.ndarray] = []
        for p in small_panels:
            parts.append(p); parts.append(sep)
        row = np.hstack(parts[:-1])
        panels.append(row)

    # 縦結合
    sep = np.full((10, panels[0].shape[1], 3), 30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts.append(p); parts.append(sep)
    grid = np.vstack(parts[:-1])
    cv2.imwrite(str(OUT_GRID), grid)
    print(f"grid: {OUT_GRID} (shape={grid.shape})")
    print(f"individual frames: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
