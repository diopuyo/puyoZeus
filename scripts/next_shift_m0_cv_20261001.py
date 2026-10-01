"""補正 queue 入力で T03 と同じ M0 の入れ子CV (3seed×5外側×(外側1+内側5)) を回す (2026-10-01)。

元手順: 56dc scripts/r2_grouped_cv_148_20260925.py の run_folds / fold_predictions / fit_predict
(fold 分割・seed・学習設定・較正は同一。火力融合 B は G_fe に不要なので作らず、predictions.csv の B は A を写す)。
出力構造は元と同じ: <root>/seed_s/fold_f/{outer_m0,inner_i_m0}.{npz,json,pt}, predictions.csv
--check は台の確認: 元入力で seed0 fold0 inner_0 を1本だけ学習し、既存 npz と確率を比べる。
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from scripts import train_advantage_m1_causal_ledger_v1 as trainer
from src.event_provisional_oof_v1 import FoldPlan

N_FOLDS, SEEDS, GPU_FRACTION = 5, (0, 1, 2), 0.85
REFERENCE = Path("/mnt/c/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer/logs/r2_grouped_cv_148_20260925")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """読み終えてから閉じる。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load(root: Path) -> tuple[dict[str, np.ndarray], trainer.CanonicalSamples]:
    """T03 prepare_data の結合と canonical (試合等重み) を再現する。"""
    chunks = [read_npz(p) for p in sorted((root / "inputs").glob("video_*.npz"))]
    data = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    _, inverse, counts = np.unique(data["game_keys"], return_inverse=True, return_counts=True)
    data["weights"] = (1.0 / counts[inverse]).astype(np.float32)
    n = len(data["labels"])
    samples = trainer.CanonicalSamples(data["boards"], data["queues"],
        np.empty((n, 2, 0), dtype=np.float32), np.empty((n, 2, 0, 5), dtype=np.float32),
        data["labels"], data["source_groups"], data["game_keys"], data["weights"],
        np.zeros(n, dtype=np.int8))
    return data, samples


def training_args() -> argparse.Namespace:
    """T03 と同じ既定学習設定。"""
    return argparse.Namespace(epochs=trainer.DEFAULT_EPOCHS, patience=trainer.DEFAULT_PATIENCE,
        batch_size=trainer.DEFAULT_BATCH_SIZE, learning_rate=trainer.DEFAULT_LEARNING_RATE,
        weight_decay=trainer.DEFAULT_WEIGHT_DECAY)


def fit_predict(samples: trainer.CanonicalSamples, train: np.ndarray, evaluation: np.ndarray,
                path: Path, seed: int) -> np.ndarray:
    """T03 fit_predict と同一 (保存済みなら再用)。"""
    if path.with_suffix(".npz").exists():
        saved = read_npz(path.with_suffix(".npz"))
        assert np.array_equal(saved["indices"], evaluation)
        return saved["probability"]
    started = time.monotonic()
    groups = samples.source_groups[train]
    fit, tune = next(GroupKFold(min(N_FOLDS, len(np.unique(groups)))).split(train, samples.labels[train], groups))
    fitting, tuning = train[fit], train[tune]
    local = samples.subset(np.concatenate((fitting, tuning, evaluation)))
    folds = np.concatenate((np.zeros(len(fitting)), np.ones(len(tuning)), np.full(len(evaluation), 2)))
    local = replace(local, folds=folds.astype(np.int8))
    args, device = training_args(), torch.device("cuda:0")
    model, slope, epoch, _, _ = trainer._fit_fold(local, FoldPlan(2, 1, (0,)), "m0", seed, args, device)
    raw = trainer._predict(model, samples.subset(evaluation), args.batch_size, device, "m0")
    probability = trainer.calibrate(raw, slope)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    slope=slope, seed=seed, best_epoch=epoch), path.with_suffix(".pt"))
    np.savez_compressed(path.with_suffix(".npz"), indices=evaluation, probability=probability)
    path.with_suffix(".json").write_text(json.dumps(dict(seconds=time.monotonic() - started, seed=seed,
        best_epoch=epoch, slope=slope)), encoding="utf-8")
    print(json.dumps(dict(event="M0_FIT", path=str(path), seconds=time.monotonic() - started)), flush=True)
    del model, local
    torch.cuda.empty_cache()
    return probability


def fold(samples: trainer.CanonicalSamples, data: dict, train: np.ndarray, test: np.ndarray,
         unit: Path, seed: int, fold_index: int) -> None:
    """外側1本と内側5本を学習し、predictions.csv を元と同じ列で保存する。"""
    base = seed * 1000 + fold_index * 10
    outer = fit_predict(samples, train, test, unit / "outer_m0", base)
    splitter = GroupKFold(N_FOLDS)
    for inner, (fitting, holdout) in enumerate(splitter.split(train, samples.labels[train],
                                                               samples.source_groups[train])):
        fit_predict(samples, train[fitting], train[holdout], unit / f"inner_{inner}_m0", base + inner + 1)
    frame = pd.DataFrame(dict(row_id=test, video_id=data["source_groups"][test], game_id=data["game_keys"][test],
        t_sec=data["t_sec"][test], frame=data["frame"][test], source_side=data["source_side"][test],
        phase=data["phase"][test], label=data["labels"][test], A=outer, B=outer, seed=seed, fold=fold_index))
    frame.to_csv(unit / "predictions.csv.tmp", index=False)
    (unit / "predictions.csv.tmp").replace(unit / "predictions.csv")


def run(root: Path, seeds: tuple[int, ...]) -> None:
    """seed ごとに StratifiedGroupKFold(shuffle, random_state=seed) で外側 fold を回す。"""
    data, samples = load(root)
    for seed in seeds:
        splits = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(
            samples.labels, samples.labels, samples.source_groups)
        for index, (train, test) in enumerate(splits):
            unit = root / f"seed_{seed}" / f"fold_{index}"
            if (unit / "predictions.csv").exists():
                continue
            fold(samples, data, train, test, unit, seed, index)
            print(json.dumps(dict(event="FOLD_DONE", seed=seed, fold=index)), flush=True)


def check(root: Path) -> None:
    """台の確認: 元入力で seed0 fold0 inner_0 を学習し、既存の確率と比べる。"""
    _, samples = load(REFERENCE)
    train, _ = next(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0).split(
        samples.labels, samples.labels, samples.source_groups))
    fitting, holdout = next(GroupKFold(N_FOLDS).split(train, samples.labels[train], samples.source_groups[train]))
    path = root / "check" / "inner_0_m0"
    got = fit_predict(samples, train[fitting], train[holdout], path, 1)
    ref = read_npz(REFERENCE / "seed_0/fold_0/inner_0_m0.npz")
    result = dict(indices_equal=bool(np.array_equal(ref["indices"], train[holdout])),
                  identical=bool(np.array_equal(ref["probability"], got)),
                  max_abs_diff=float(np.max(np.abs(ref["probability"] - got))), rows=int(len(got)))
    (root / "check" / "CHECK.json").write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main() -> None:
    """GPU 1本。--check は台の確認だけ。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
    if args.check:
        check(args.root)
        return
    run(args.root, tuple(args.seeds))


if __name__ == "__main__":
    main()
