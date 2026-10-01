"""補正 queue の M0 (T03 再実行結果) で G_fe を元と同じ手順で CV・最終学習する (2026-10-01)。

元手順: scripts/train_exchange_event_models_20260926.py (E1) の fit_g_fold / final_fit の G_fe 部分。
差し替えるのは M0 の OOF (run_path(reference, "m0")) と rows.A (seed0 外側 OOF) だけ。D・位相・行・fold は元と同一。
参照ディレクトリ構造を作り (元資産はシンボリックリンク、rows.csv だけ A を差し替え)、v1 の関数をそのまま呼ぶ。
出力: <output>/seed_s/fold_f/g.csv、<output>/G_CV.json、--models に G_fe.joblib と manifest.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from scripts import train_exchange_event_models_20260926 as v1
from src.exchange_event_features import PHASE_BOUNDS, g_features

ORIGINAL_CV = Path("/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/exchange_event_v1")
ORIGINAL_MODELS = Path("models/exchange_event_v1")
LINKED = ("g", "s1", "s3")
WORKERS = 5


def build_reference(m0_root: Path, reference: Path) -> None:
    """元参照の構造を複製し、m0 と base/rows.csv の A だけ新しいものにする。"""
    logs = reference / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for kind in LINKED:
        link = logs / v1.RUN_NAMES[kind]
        if not link.exists():
            link.symlink_to(v1.run_path(v1.REFERENCE, kind))
    m0 = logs / v1.RUN_NAMES["m0"]
    if not m0.exists():
        m0.symlink_to(m0_root.resolve())
    base, source = logs / v1.RUN_NAMES["base"], v1.run_path(v1.REFERENCE, "base")
    base.mkdir(exist_ok=True)
    for name in ("design.npy", "columns.json"):
        if not (base / name).exists():
            (base / name).symlink_to(source / name)
    rows = pd.read_csv(source / "rows.csv")
    outer = pd.concat([pd.read_csv(p) for p in sorted((m0_root / "seed_0").glob("fold_*/predictions.csv"))])
    outer = outer.set_index("row_id").loc[rows.row_id]
    np.testing.assert_array_equal(outer.label.to_numpy(), rows.label.to_numpy())
    np.testing.assert_array_equal(outer.fold.to_numpy(), rows.fold.to_numpy())
    rows["A"] = outer.A.to_numpy()
    rows.to_csv(base / "rows.csv", index=False)


def g_job(job: tuple[str, str, int, int]) -> str:
    """1 fold の G_fe を元の fit_g_fold で学習して保存する。"""
    reference, output, seed, fold = job
    os.nice(max(0, 19 - os.nice(0)))
    path = Path(output) / f"seed_{seed}/fold_{fold}/g.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        v1.fit_g_fold(Path(reference), seed, fold).to_csv(path, index=False)
    return str(path)


def paired_metrics(output: Path, seeds: tuple[int, ...]) -> dict:
    """元 CV (exev logs/exchange_event_v1) と同じ行で LL/AUC を比べる (全体・中盤)。"""
    result = {}
    new = pd.concat([pd.read_csv(output / f"seed_{s}/fold_{f}/g.csv").assign(seed=s)
                     for s in seeds for f in v1.FOLDS])
    old = pd.concat([pd.read_csv(ORIGINAL_CV / f"seed_{s}/fold_{f}/g.csv").assign(seed=s)
                     for s in seeds for f in v1.FOLDS])
    merged = new.merge(old, on=["seed", "row_id"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(merged) == len(new) == len(old)
    for scope, frame in (("overall", merged), ("middle", merged[merged.phase_new == v1.MIDDLE])):
        y = frame.label_new.to_numpy()
        result[scope] = dict(rows=len(frame), **{f"{m}_{k}": f(y, frame[f"G_fe_{k}"])
            for k in ("old", "new") for m, f in (("log_loss", log_loss), ("auc", roc_auc_score))})
    return result


def final_fit(reference: Path, models: Path) -> None:
    """E1 final_fit の G_fe 部分と同一。manifest は共通側 (G_fe・M0) 用に書く。"""
    rows, design, phases = v1.datasets(reference)
    thresholds = np.quantile(phases["elapsed"], PHASE_BOUNDS)
    phase = np.column_stack((phases["fill"], np.searchsorted(thresholds, phases["elapsed"], side="left")))
    sign, label = rows.sign.to_numpy(), rows.label.to_numpy()
    model = LogisticRegression(**v1.LR_PARAMS).fit(g_features(design, rows.A.to_numpy(), sign, phase),
                                                   np.where(sign > 0, label, 1 - label))
    models.mkdir(parents=True, exist_ok=True)
    original = json.loads((ORIGINAL_MODELS / "manifest.json").read_text(encoding="utf-8"))
    assert np.allclose(thresholds, original["elapsed_thresholds"])
    entry = v1.save_model(models, "G_fe", model)
    v1.save_json(models / "manifest.json", dict(version=v1.MODEL_VERSION, elapsed_thresholds=thresholds.tolist(),
        models=dict(G_fe=entry), m0_training="補正queue(T)のT03再実行 seed0 外側OOF (rows.A)",
        queue_semantics="P_k,P_{k+1} (logs/next_shift_train, 2026-10-01)"))


def main() -> None:
    """参照構築のあと、final = 最終学習 (seed0 外側OOFだけ使う) / cv = 15fold CV と対応比較。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--stage", choices=("final", "cv"), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(v1.SEEDS))
    args = parser.parse_args()
    reference = args.output / "reference"
    build_reference(args.m0_root, reference)
    if args.stage == "final":
        final_fit(reference, args.models)
        return
    seeds = tuple(args.seeds)
    jobs = [(str(reference), str(args.output), s, f) for s in seeds for f in v1.FOLDS]
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(g_job, jobs))
    name = "G_CV.json" if seeds == tuple(v1.SEEDS) else "G_CV_seeds_" + "_".join(map(str, seeds)) + ".json"
    v1.save_json(args.output / name, dict(seeds=list(seeds), **paired_metrics(args.output, seeds)))


if __name__ == "__main__":
    main()
