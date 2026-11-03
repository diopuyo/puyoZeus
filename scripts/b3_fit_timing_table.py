"""全完了動画から時間軸テーブル (1手時間=高さ別 中央値/下位10%、段時間、最終段後の終端) を推定する。

出力は全データ推定 (fit_scope=all_data_not_fold_safe): 学習の外側 test を含むため評価には使わず、
学習時は fold 内サンプルで b3_timing.fit_timing_table を呼び直して差し替える。
サンプル本体も保存し、fold 内再推定に使えるようにする。
使い方: python -m scripts.b3_fit_timing_table --out D:/puyo_analyzer/b3_data/timing_v1
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.b3_source import iter_frames, list_complete_videos, load_events, load_lean, resolve_attempt
from src.b3_timing import collect_hand_samples, fit_timing_table

RUN_GAP_SEC = 3.0
MIN_RUN_STEPS = 3
QUIET = ("STABLE", "TSUMO_FALL", "OJAMA_FALL")
QUIET_FRAMES = 2


def _video_samples(video_id: str) -> dict:
    p = resolve_attempt(video_id)
    lean = load_lean(p.record)
    hand = collect_hand_samples(lean.grids, lean.side, lean.game, lean.t_sec)
    events = [e for e in load_events(p.record) if e["kind"] == "formula_step"]
    quiet = {"1P": [], "2P": []}
    for d in iter_frames(p.record):
        for name, s in zip(("1P", "2P"), d["sides"]):
            quiet[name].append((d["t_sec"], s["state"] in QUIET))
    gaps, tails = [], []
    for side in ("1P", "2P"):
        steps = sorted(e["t_sec"] for e in events if e["side"] == side)
        runs, cur = [], []
        for t in steps:
            if cur and t - cur[-1] > RUN_GAP_SEC:
                runs.append(cur)
                cur = []
            cur.append(t)
        runs.append(cur) if cur else None
        times = np.array([x[0] for x in quiet[side]])
        flags = np.array([x[1] for x in quiet[side]])
        for run in runs:
            if len(run) < MIN_RUN_STEPS:
                continue
            gaps += list(np.diff(run))
            i = int(np.searchsorted(times, run[-1], side="right"))
            while i + QUIET_FRAMES <= len(times) and not flags[i:i + QUIET_FRAMES].all():
                i += 1
            if i + QUIET_FRAMES <= len(times):
                tails.append(float(times[i] - run[-1]))
    return {"video": video_id, "hand": hand, "gaps": np.array(gaps), "tails": np.array(tails)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--videos", nargs="*")
    a = ap.parse_args()
    vids = a.videos or list_complete_videos()
    with Pool(a.workers) as pool:
        res = pool.map(_video_samples, vids)
    a.out.mkdir(parents=True, exist_ok=True)
    hand = np.concatenate([r["hand"] for r in res])
    gaps = np.concatenate([r["gaps"] for r in res])
    tails = np.concatenate([r["tails"] for r in res])
    np.savez_compressed(a.out / "samples.npz", hand=hand, gaps=gaps, tails=tails,
                        hand_video=np.concatenate([[r["video"]] * len(r["hand"]) for r in res]).astype(str))
    table = fit_timing_table(hand, gaps, tails, meta={"videos": vids})
    (a.out / "timing_table.json").write_text(table.to_json(), encoding="utf-8")
    print(table.to_json())


if __name__ == "__main__":
    main()
