"""段2のV3/V1bだけを全採用行で最終学習する。既存モデルは読取り専用。"""
from __future__ import annotations

import os
import resource
import time
from pathlib import Path

from scripts import gfe_phase_stage1_20261002 as stage
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.exchange_event_evaluator import file_sha256
from src.exchange_event_user_phase import USER_COLUMNS, USER_MODEL_VERSION

OUT = stage.ROOT / "logs/gfe_phase_stage2_20261002"
VARIANTS = ("V3", "V1b")


def model_directory(variant: str) -> Path:
    """候補専用ディレクトリを返す。既存v1のパスを使わない。"""
    return stage.ROOT / f"models/exchange_event_v1_userphase_{variant.lower()}"


def fit(variant: str, rows: pd.DataFrame, design: np.ndarray, u: np.ndarray,
        missing: np.ndarray) -> dict:
    """既存最終学習と同じseed0のM0外側OOFを全行で用いる。"""
    stage.memory_guard()
    started = time.monotonic()
    directory = model_directory(variant)
    directory.mkdir(parents=True, exist_ok=False)
    sign, labels = rows.sign.to_numpy(), rows.label.to_numpy()
    x = stage.features(variant, design, rows.A.to_numpy(), sign,
                       np.column_stack((u, np.zeros_like(u))), missing)
    model = (HistGradientBoostingClassifier(**stage.HGB_PARAMS) if variant == "V3" else
             LogisticRegression(**stage.LR_PARAMS))
    model.fit(x, np.where(sign > 0, labels, 1-labels))
    path = directory / "G_fe.joblib"
    joblib.dump(model, path)
    entry = dict(file=path.name, sha256=file_sha256(path), columns=list(USER_COLUMNS),
                 estimator_type=type(model).__name__, parameters=model.get_params())
    meta = dict(version=USER_MODEL_VERSION, variant=variant, models=dict(G_fe=entry),
        rows=len(rows), games=int(rows.game_id.nunique()), videos=int(rows.video_id.nunique()),
        m0="seed0 outer OOF: baseline rows.A（既存final_fitと同じ）", sklearn_version=sklearn.__version__,
        seconds=time.monotonic()-started, adopted=False, optional_default=False)
    stage.save_json(directory / "manifest.json", meta)
    stage.save_json(OUT / f"TRAIN_{variant}.json", meta)
    print(f"{variant} 最終学習完了 {meta['seconds']:.2f}秒", flush=True)
    return meta


def main() -> None:
    """資源条件を確認してV3、V1bを順番に保存する。"""
    os.nice(max(0, 19-os.nice(0)))
    resource.setrlimit(resource.RLIMIT_AS, (stage.MAX_MEMORY, stage.MAX_MEMORY))
    stage.memory_guard()
    rows, design, _ = stage.datasets(stage.REFERENCE)
    with np.load(stage.OUTPUT / "user_phase.npz") as saved:
        u, missing = saved["phase"], saved["missing"]
    for variant in VARIANTS:
        fit(variant, rows, design, u, missing)


if __name__ == "__main__":
    main()
