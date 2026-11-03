"""合成データだけで HGB のスレッド数別の決定性と所要時間を測る。

WSL でリポジトリを cwd にして PYTHONPATH=. python <このファイル> <出力JSON> <PC名>。
share/data/equiv/determinism_threads.py に同じスクリプトを配布する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import sklearn

from src import b3_train_model
from src.b3_train_model import DEFAULT_GRID, HgbConfig, feature_columns, fit_two_state
from src.b3_train_synth import make_rows

SEED = 1003
N_VIDEOS, GAMES, ROWS, TRAIN_VIDEOS = 20, 5, 200, 15
THREAD_RUNS = (1, 8, 8)


def measure(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], cfg: HgbConfig) -> dict:
    """学習と予測を計測し、予測の生バイトを比較する (採点はしない)。"""
    predictions, runs = [], []
    for threads in THREAD_RUNS:
        os.environ["B3_HGB_THREADS"] = str(threads)
        start = time.perf_counter()
        model = fit_two_state(train, columns, cfg, SEED)
        p = model.predict(test).astype(np.float64)
        elapsed = time.perf_counter() - start
        predictions.append(p)
        runs.append({"hgb_threads": threads, "blas_threads": 1, "seconds": elapsed,
                     "pred_sha256": hashlib.sha256(p.tobytes()).hexdigest()})
        print(f"合成データ: {cfg.name}, {threads} スレッド, {elapsed:.3f} 秒", flush=True)
    one, eight_a, eight_b = predictions
    return {"params": asdict(cfg), "runs": runs,
            "eight_repeat_bitwise_equal": eight_a.tobytes() == eight_b.tobytes(),
            "one_vs_eight_bitwise_equal": one.tobytes() == eight_a.tobytes(),
            "one_vs_eight_max_abs_diff": float(np.max(np.abs(one - eight_a))),
            "eight_repeat_max_abs_diff": float(np.max(np.abs(eight_a - eight_b))),
            "speedup_one_over_eight": runs[0]["seconds"] / np.mean([r["seconds"] for r in runs[1:]])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, help="出力 JSON のパス")
    ap.add_argument("pc", help="PC の識別名")
    args = ap.parse_args()
    df, _ = make_rows(n_videos=N_VIDEOS, games_per_video=GAMES, rows_per_game=ROWS, seed=SEED)
    vids = sorted(df["video_id"].unique())
    train = df[df["video_id"].isin(vids[:TRAIN_VIDEOS])]
    test = df[~df["video_id"].isin(vids[:TRAIN_VIDEOS])].sort_values("row_key")
    columns = feature_columns(df.columns)
    result = {"pc": args.pc, "python": sys.version, "platform": platform.platform(),
              "numpy": np.__version__, "sklearn": sklearn.__version__, "seed": SEED,
              "data_sha256": hashlib.sha256(df.sort_values("row_key").to_csv(index=False).encode()).hexdigest(),
              "n_train": len(train), "n_test": len(test), "n_columns": len(columns), "configs": {}}
    for cfg in DEFAULT_GRID:
        result["configs"][cfg.name] = measure(train, test, columns, cfg)
        print(f"合成データ: {cfg.name} 完了", flush=True)
    configs = list(result["configs"].values())
    result["eight_repeat_bitwise_equal"] = all(c["eight_repeat_bitwise_equal"] for c in configs)
    result["one_vs_eight_max_abs_diff"] = max(c["one_vs_eight_max_abs_diff"] for c in configs)
    result["seconds_one"] = sum(c["runs"][0]["seconds"] for c in configs)
    result["seconds_eight_mean"] = sum(np.mean([r["seconds"] for r in c["runs"][1:]]) for c in configs)
    result["speedup_one_over_eight"] = result["seconds_one"] / result["seconds_eight_mean"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result["code_head"] = git.stdout.strip() if git.returncode == 0 else None
    result["model_source_sha256"] = hashlib.sha256(Path(b3_train_model.__file__).read_bytes()).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
