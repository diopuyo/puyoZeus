"""本番M0レシピを再利用する、外側train内だけのnested OOF。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch

from scripts import train_advantage_m1_causal_ledger_v1 as recipe
from src.event_provisional_oof_v1 import FoldPlan
from src.exchange_event_m0 import LOGIT_LIMIT

TUNE_FOLDS = 5
INNER_FOLDS = 3
DEFAULT_SEED = 0
DEFAULT_ARGS = dict(epochs=12, patience=3, batch_size=512, learning_rate=3e-4, weight_decay=1e-4)


def canonical(df: pd.DataFrame, boards: np.ndarray, queues: np.ndarray,
              groups: dict[str, str]) -> recipe.CanonicalSamples:
    """本番と同じ最終勝敗・試合均等重み。B3の左右複製を含め試合総重み1。"""
    if df.row_key.duplicated().any() or not df.won_p1.isin([0, 1]).all():
        raise ValueError("行キー重複または教師欠測")
    if not df.video_id.isin(groups).all():
        raise ValueError("事前登録に無い動画")
    games = df.video_id.astype(str) + ":" + df.game_idx.astype(str)
    weights = 1 / games.map(games.value_counts()).to_numpy(dtype=np.float32)
    n = len(df)
    return recipe.CanonicalSamples(boards, queues, np.empty((n, 2, 0), np.float32),
        np.empty((n, 2, 0, 5), np.float32), df.won_p1.to_numpy(dtype=np.float32),
        df.video_id.map(groups).to_numpy(dtype=str), games.to_numpy(dtype=str), weights,
        np.zeros(n, dtype=np.int8))


def tune_split(samples: recipe.CanonicalSamples, smoke: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """通常は動画groupでfit/tune分離。3動画試走の内側だけ試合分離を許す。"""
    groups, policy = samples.source_groups, "source_group"
    if len(np.unique(groups)) < 2 and smoke:
        groups, policy = samples.game_keys, "smoke_only_game"
    count = min(TUNE_FOLDS, len(np.unique(groups)))
    if count < 2:
        raise ValueError("fit/tuneを分離するgroupが不足")
    fit, tune = next(GroupKFold(count).split(samples.labels, groups=groups))
    return fit, tune, policy


def production_logits(probabilities: np.ndarray, slope: float) -> np.ndarray:
    """FileM0Predictorの対称較正を経た確率をG_feと同じlogitへ戻す。"""
    z = np.clip(recipe._logits(probabilities) * slope, -LOGIT_LIMIT, LOGIT_LIMIT)
    probability = 1 / (1 + np.exp(-z))
    epsilon = np.finfo(np.float64).eps
    probability = np.clip(probability, epsilon, 1 - epsilon)
    return np.log(probability) - np.log1p(-probability)


def fit_predict(train: recipe.CanonicalSamples, test: recipe.CanonicalSamples, output: Path,
                args: argparse.Namespace, device: torch.device, seed: int, smoke: bool) -> tuple[np.ndarray, dict]:
    """評価側ラベルを学習・epoch選択・較正に渡さない。成績の出力もしない。"""
    if set(train.game_keys) & set(test.game_keys) or set(train.source_groups) & set(test.source_groups):
        raise ValueError("M0の学習側と予測側が重複")
    fit, tune, policy = tune_split(train, smoke)
    folds = np.zeros(len(train.labels), dtype=np.int8)
    folds[tune] = 1
    started = time.monotonic()
    # 旧レシピの学習ログは非表示。epoch選択と較正はtrain内tuneだけで完結する。
    with redirect_stdout(io.StringIO()) as captured:
        model, slope, epoch, _, _ = recipe._fit_fold(
            replace(train, folds=folds), FoldPlan(2, 1, (0,)), "m0", seed, args, device)
    completed_epochs = captured.getvalue().count("epoch=")
    probability = recipe._predict(model, test, args.batch_size, device, "m0")
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "slope": slope, "seed": seed, "best_epoch": epoch}, output)
    info = dict(seconds=time.monotonic() - started, fit_rows=len(fit), tune_rows=len(tune),
                predict_rows=len(test.labels), epochs_run=completed_epochs, best_epoch=epoch,
                tune_policy=policy, fit_groups=sorted(set(train.source_groups[fit])),
                tune_groups=sorted(set(train.source_groups[tune])),
                predict_groups=sorted(set(test.source_groups)), checkpoint=output.name)
    del model
    return production_logits(probability, slope), info


def nested_fold(samples: recipe.CanonicalSamples, fold_rows: np.ndarray, fold: int, output: Path,
                args: argparse.Namespace, device: torch.device, seed: int = DEFAULT_SEED,
                smoke: bool = False) -> tuple[np.ndarray, list[dict]]:
    """外側testのOOFと、その外側train専用の内側OOFを1組で作る。"""
    train_idx, test_idx = np.flatnonzero(fold_rows != fold), np.flatnonzero(fold_rows == fold)
    if not len(train_idx) or not len(test_idx):
        raise ValueError("外側train/testが空")
    train = samples.subset(train_idx)
    count = min(INNER_FOLDS, len(np.unique(train.source_groups)))
    if count < 2:
        raise ValueError("内側OOFには外側trainが2group以上必要")
    values, records = np.full(len(samples.labels), np.nan), []
    splits = GroupKFold(count).split(train.labels, groups=train.source_groups)
    for inner, (fit, held) in enumerate(splits):
        values[train_idx[held]], info = fit_predict(train.subset(fit), train.subset(held),
            output / f"fold{fold}_inner{inner}.pt", args, device, seed, smoke)
        records.append({"inner_fold": inner, **info})
    values[test_idx], info = fit_predict(train, samples.subset(test_idx),
        output / f"fold{fold}_outer.pt", args, device, seed, smoke)
    records.append({"inner_fold": None, **info})
    if not np.isfinite(values).all():
        raise ValueError("M0 logitが未予測または非有限")
    return values, records
