"""T03の144動画入力・動画内側splitで本番用M0を最終学習する。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from scripts import train_advantage_m1_causal_ledger_v1 as trainer
from src.event_provisional_oof_v1 import FoldPlan
from src.exchange_event_evaluator import file_sha256

GPU_FRACTION = .85
EXPECTED_VIDEOS = 144
TUNE_FOLDS = 5
SEED = 0


def load_inputs(root: Path) -> trainer.CanonicalSamples:
    """T03で採用済みの入力を読み取り専用で再用する。"""
    paths = sorted(root.glob("video_*.npz"))
    assert len(paths) == EXPECTED_VIDEOS, len(paths)
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            chunks.append({key: source[key] for key in source.files})
    data = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    n = len(data["labels"])
    _, inverse, counts = np.unique(data["game_keys"], return_inverse=True, return_counts=True)
    return trainer.CanonicalSamples(data["boards"], data["queues"],
        np.empty((n, 2, 0), dtype=np.float32), np.empty((n, 2, 0, 5), dtype=np.float32),
        data["labels"], data["source_groups"], data["game_keys"],
        (1 / counts[inverse]).astype(np.float32), np.zeros(n, dtype=np.int8))


def train(inputs: Path, output: Path) -> None:
    """全採用動画からfit/tuneを分離し、最良epochと較正を保存する。"""
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    samples = load_inputs(inputs)
    fit, tune = next(GroupKFold(TUNE_FOLDS).split(
        samples.labels, samples.labels, samples.source_groups))
    folds = np.zeros(len(samples.labels), dtype=np.int8)
    folds[tune] = 1
    samples = replace(samples, folds=folds)
    args = argparse.Namespace(epochs=12, patience=3, batch_size=512,
                              learning_rate=3e-4, weight_decay=1e-4)
    print(f"rows={len(folds)} fit={len(fit)} tune={len(tune)}", flush=True)
    model, slope, epoch, raw, calibrated = trainer._fit_fold(
        samples, FoldPlan(2, 1, (0,)), "m0", SEED, args, torch.device("cuda:0"))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    slope=slope, seed=SEED, best_epoch=epoch), checkpoint)
    metadata = dict(model="AdvantageM0CurrentCNNV2", hyperparameters=vars(args), seed=SEED,
        seconds=time.monotonic() - started, best_epoch=epoch, slope=slope,
        tune_raw=raw, tune_calibrated=calibrated, rows=len(folds), videos=EXPECTED_VIDEOS,
        fit_videos=np.unique(samples.source_groups[fit]).tolist(),
        tune_videos=np.unique(samples.source_groups[tune]).tolist(),
        peak_vram_bytes=torch.cuda.max_memory_allocated(), gpu_fraction=GPU_FRACTION,
        checkpoint_sha256=file_sha256(checkpoint), input_root=str(inputs))
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


def main() -> None:
    """保存先は作業ツリー内、検証資産は入力専用とする。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/exchange_event_v1/M0"))
    args = parser.parse_args()
    train(args.inputs, args.output)


if __name__ == "__main__":
    main()
