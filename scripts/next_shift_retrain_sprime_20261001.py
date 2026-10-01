"""補正 queue で v3 (E15 live count) の S1′/S3′ を同一手順で再学習する (2026-10-01)。

元手順: scripts/train_exchange_event_models_v3_20260928.py (行・fold・seed・重み・パラメータは不変)。
差し替えるのは特徴計算で読む queue (v2.queue_at) だけ。行の採否 (samples の変化検出) は原票の queue のまま
にして、行集合を元と一対一に保つ。
使い方:
  PYTHONPATH=. python -m scripts.next_shift_retrain_sprime_20261001 --variant T --output logs/next_shift_train/sprime_T
  --variant orig は台の確認 (元の特徴と一致するか) 用。--videos で動画を絞れる (台の確認用)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import train_exchange_event_models_v3_20260928 as v3

v2, v1 = v3.v2, v3.v1
QUEUES = Path("logs/next_shift_train/queues")
VARIANTS = ("orig", "C", "T")
FIXED_KEY = "queue_fixed"
STATE: dict[str, str] = {}


def fixed_queue_at(raw: dict, idx: int) -> tuple[int, ...]:
    """v2.queue_at と同じ未知色処理で、補正 queue を読む。"""
    return tuple(int(v) if 1 <= v <= 5 else 0 for v in raw[FIXED_KEY][idx])


def load_raw(video: str) -> dict:
    """原票に補正 queue 列を足す (原票の queue 列は変化検出用にそのまま残す)。"""
    raw = v2.read_npz(v2.RAW / (video.removeprefix("video_") + ".npz"))
    with np.load(QUEUES / (video.removeprefix("video_") + ".npz")) as fixed:
        np.testing.assert_array_equal(fixed["queue_orig"], np.stack([raw[k] for k in v2.QUEUE], 1))
        raw[FIXED_KEY] = fixed["queue_" + STATE["variant"]]
    return raw


def video_job(job: tuple[str, str]) -> str:
    """v3.video_job と同一。原票の読込だけ load_raw に替える。"""
    video, output = job
    root = Path(output) / "features" / video
    raw = load_raw(video)
    parts, design = v3.reference_inputs()
    rows = parts.get(video, parts[""])
    for game in np.unique(raw["game_idx"]):
        path = root / f"game_{game}.npz"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            continue
        ids = np.flatnonzero(raw["game_idx"] == game)
        ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
        part = rows[rows.game_id == f"{video}:{game}"]
        values = list(v3.game_rows(raw, ids, part, design))
        info = [dict(video_id=video, **v[0]) for v in values]
        v2.save_npz(path, s1=np.asarray([v[1] for v in values], dtype=np.float32),
                    s3=np.asarray([v[2] for v in values], dtype=np.float32))
        v1.save_json(meta, info)
    print(f"特徴完了 {video}", flush=True)
    return video


def install(variant: str, models: Path | None) -> None:
    """fork 前に差し替える (子プロセスへ継承される)。"""
    STATE["variant"] = variant
    v2.queue_at = fixed_queue_at
    v3.video_job = video_job
    if models is not None:
        v3.MODELS = models


def features_only(output: Path, videos: list[str]) -> None:
    """台の確認用: 指定動画の特徴だけ作る。"""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(v3.WORKERS, len(videos)), initializer=v2.init_worker) as pool:
        list(pool.map(video_job, [(v, str(output)) for v in videos]))


def full_run(output: Path) -> None:
    """v3.main と同じ段取り (特徴→15fold CV→報告→最終学習)。"""
    from concurrent.futures import ProcessPoolExecutor
    v1.save_json(output / "STATUS.json", dict(stage="FEATURES"))
    v3.prepare(output)
    v1.save_json(output / "STATUS.json", dict(stage="CV"))
    with ProcessPoolExecutor(v3.WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(v3.fold_job, [(str(output), s, f) for s in v1.SEEDS for f in v1.FOLDS]))
    report = v3.cv_report(output)
    v1.save_json(output / "METRICS.json", report)
    if not report["a"]["leak_suspected"]:
        v3.final_fit(output)
    v1.save_json(output / "STATUS.json", dict(stage=report["status"]))


def main() -> None:
    """引数を読み、差し替えを入れてから実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=v3.WORKERS)
    args = parser.parse_args()
    v3.WORKERS = args.workers
    install(args.variant, args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    v1.save_json(args.output / "RUN.json", dict(variant=args.variant, videos=args.videos,
                                               models=str(args.models), workers=args.workers))
    if args.videos:
        features_only(args.output, args.videos)
        return
    if args.models is None:
        raise ValueError("全体実行には --models (新しい保存先) が必要")
    full_run(args.output)


if __name__ == "__main__":
    main()
