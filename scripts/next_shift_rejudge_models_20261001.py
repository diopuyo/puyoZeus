"""再判定 (exev DECISIONS 2026-10-01 事前登録 (再判定)) 用に、旧/補正の各 random_state のモデル一式を作る。

構成 <variant>_rs<k> (variant = orig / T、k = 0..4):
- models/rejudge/<variant>_rs<k>/        S1′/S3′(+light) を v3.final_fit と同じ手順・random_state=k で学習
- models/rejudge/<variant>_rs<k>_common/ G_fe (E1 と同じ LR、random_state=k) + M0 (train_exchange_event_m0、seed=k)
元特徴: orig = exev logs/e15 (現本番 v3 の学習行列)、T = logs/next_shift_train/sprime_T。
G_fe の M0 入力 (T03 seed0 外側OOF) は元の行のまま: orig = 56dc 参照、T = logs/next_shift_train/gfe_T/reference。
使い方: python -m scripts.next_shift_rejudge_models_20261001 --variant T --seed 1 --part sprime|gfe|m0
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# 学習スクリプト (v2/E1) は import 時に CUDA_VISIBLE_DEVICES を空にするので、M0 (GPU) の部品では読まない。
ROOT = Path("models/rejudge")
FEATURES = dict(orig=Path("/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15"),
                T=Path("logs/next_shift_train/sprime_T"))
G_REFERENCE = dict(orig=Path("/mnt/c/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer"),
                   T=Path("logs/next_shift_train/gfe_T/reference"))
M0_INPUTS = dict(orig=Path("/mnt/c/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer/logs/r2_grouped_cv_148_20260925/inputs"),
                 T=Path("logs/next_shift_train/m0_T/inputs"))


def name(variant: str, seed: int) -> str:
    """構成名。"""
    return f"{variant}_rs{seed}"


def sprime(variant: str, seed: int) -> None:
    """v3.final_fit と同じ学習を random_state=seed で行う。"""
    from scripts import train_exchange_event_models_v3_20260928 as v3
    from scripts import train_exchange_event_models_20260926 as e1
    v2 = v3.v2
    out = ROOT / name(variant, seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = v2.pd.read_csv(FEATURES[variant] / "rows.csv")
    target = np.where(rows.sign > 0, rows.label, 1 - rows.label)
    manifest = dict(models={}, production_enabled=False, version="E15_live_count",
                    shared_directory=name(variant, seed) + "_common", rejudge=dict(variant=variant, random_state=seed))
    for stage, columns in (("S1", v2.S1_COLUMNS), ("S3", v2.S3_COLUMNS)):
        x = np.load(FEATURES[variant] / (stage + "_prime.npy"), mmap_mode="r")
        for suffix in ("", "_light"):
            key = stage + "_prime" + suffix
            model = v2.HistGradientBoostingClassifier(random_state=seed, **(v3.v1.LIGHT_PARAMS if suffix else {}))
            model.fit(x, target, sample_weight=rows.weight)
            path = out / (key + ".joblib")
            v2.joblib.dump(model, path)
            manifest["models"][key] = dict(file=path.name, sha256=e1.file_sha256(path),
                columns=columns + v2.NEW_COLUMNS, valid=True, version="E15_live_count")
    e1.save_json(out / "manifest.json", manifest)


def gfe(variant: str, seed: int) -> None:
    """E1 final_fit の G_fe 部分 (LR、random_state=seed)。"""
    from sklearn.linear_model import LogisticRegression
    from scripts import train_exchange_event_models_20260926 as e1
    from src.exchange_event_features import PHASE_BOUNDS, g_features
    assert e1.REFERENCE == G_REFERENCE["orig"]
    out = ROOT / (name(variant, seed) + "_common")
    out.mkdir(parents=True, exist_ok=True)
    rows, design, phases = e1.datasets(G_REFERENCE[variant])
    thresholds = np.quantile(phases["elapsed"], PHASE_BOUNDS)
    phase = np.column_stack((phases["fill"], np.searchsorted(thresholds, phases["elapsed"], side="left")))
    sign, label = rows.sign.to_numpy(), rows.label.to_numpy()
    params = dict(e1.LR_PARAMS, random_state=seed)
    model = LogisticRegression(**params).fit(g_features(design, rows.A.to_numpy(), sign, phase),
                                             np.where(sign > 0, label, 1 - label))
    entry = e1.save_model(out, "G_fe", model)
    e1.save_json(out / "manifest.json", dict(version=e1.MODEL_VERSION, elapsed_thresholds=thresholds.tolist(),
        models=dict(G_fe=entry), rejudge=dict(variant=variant, random_state=seed)))


def m0(variant: str, seed: int) -> None:
    """train_exchange_event_m0 と同じ学習を seed だけ変えて行う (seed0 は既存資産を写す)。"""
    from scripts import train_exchange_event_m0 as trainer
    out = ROOT / (name(variant, seed) + "_common") / "M0"
    if (out / "manifest.json").exists():
        return
    if seed == 0:
        source = Path("models/exchange_event_v1/M0" if variant == "orig" else "models/exchange_event_v5_common/M0")
        shutil.copytree(source, out)
        return
    trainer.SEED = seed
    trainer.train(M0_INPUTS[variant], out)


def main() -> None:
    """部品ごとに実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("orig", "T"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--part", choices=("sprime", "gfe", "m0"), required=True)
    args = parser.parse_args()
    dict(sprime=sprime, gfe=gfe, m0=m0)[args.part](args.variant, args.seed)
    print(json.dumps(vars(args)), flush=True)


if __name__ == "__main__":
    main()
