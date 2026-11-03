"""148動画の識別可能な着弾区間で、動画単位15foldの応手確率を学習する。"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scripts import e19_response_training_rows as rows_builder
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.landing_counter_probability import COLUMNS, MODEL_PATH

OUT = rows_builder.OUT
FOLDS, WORKERS, MAX_ITER = 15, 3, 1000
EVALUATION_IDS = ("q_7gc4TgFig", "fcXG83vInDY", "mia8KCjr52g", "c0BQoMJwwQU")


def inputs() -> list[Path]:
    """148動画の上級ティアと評価動画との非重複を、URLのIDでも確認する。"""
    data = rows_builder.data
    paths = sorted(data.RAW.glob("*.npz"))
    assert len(paths) == 148
    tiers = pd.read_csv(data.SHARED/"data/video_tier_index_2026-08-07.tsv", sep="\t").set_index("video_name")
    accepted, excluded = [], []
    for path in paths:
        video = "video_"+path.stem
        if video not in tiers.index or not any(t in str(tiers.loc[video, "tier"]) for t in data.v1.ALLOWED_TIERS):
            raise ValueError("ティア未確認: "+video)
        if str(tiers.loc[video, "video_id"]) in EVALUATION_IDS:
            excluded.append(video)
        else:
            accepted.append(path)
    save_json(OUT/"INPUTS.json", dict(input_videos=len(paths), eligible_videos=len(accepted),
        excluded_evaluation=excluded, evaluation_ids=EVALUATION_IDS,
        raw=str(data.RAW), videos=[p.stem for p in accepted]))
    return accepted


def prepare(paths: list[Path]) -> pd.DataFrame:
    """未来の識別不能区間を陰性にせず、母数と除外数を保存する。"""
    with ProcessPoolExecutor(WORKERS, initializer=rows_builder.data.init_worker) as pool:
        chunks = list(pool.map(rows_builder.video_job, map(str, paths)))
    counts = Counter()
    records = []
    for chunk in chunks:
        counts.update(chunk["counts"])
        records.extend(chunk["rows"])
    rows = pd.DataFrame(records)
    rows.to_json(OUT/"candidates.jsonl", orient="records", lines=True)
    labeled = rows[rows.label.notna()].copy().reset_index(drop=True)
    for i, name in enumerate(COLUMNS):
        labeled[name] = labeled.features.map(lambda x: x[i])
    labeled.drop(columns="features").to_csv(OUT/"rows.csv", index=False)
    save_json(OUT/"COHORT.json", dict(counts=dict(counts), candidates=len(rows), rows=len(labeled),
        videos=int(labeled.video_id.nunique()), positives=int(labeled.label.sum()),
        label_source="おじゃま増加の前後観測区間。実落下通知の代用であり、未観測の落下は復元不能",
        excluded=int(rows.label.isna().sum())))
    return labeled


def fit(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """標準化も学習動画だけで推定し、クラス重みで発生確率を歪めない。"""
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1., max_iter=MAX_ITER, random_state=0).fit(scaler.transform(x), y)
    return scaler, model


def train(rows: pd.DataFrame) -> None:
    """動画非重複15foldのOOFと全採用動画fitを別々に保存する。"""
    x, y = rows[list(COLUMNS)].to_numpy(float), rows.label.to_numpy(int)
    assert np.isfinite(x).all() and rows.video_id.nunique() >= FOLDS
    predictions, constant = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    folds, assignments = [], np.full(len(rows), -1)
    for number, (train_ids, test_ids) in enumerate(GroupKFold(FOLDS).split(x, y, rows.video_id)):
        train_v, test_v = set(rows.iloc[train_ids].video_id), set(rows.iloc[test_ids].video_id)
        assert not train_v & test_v
        scaler, model = fit(x[train_ids], y[train_ids])
        predictions[test_ids] = model.predict_proba(scaler.transform(x[test_ids]))[:, 1]
        constant[test_ids], assignments[test_ids] = y[train_ids].mean(), number
        folds.append(dict(fold=number, train_videos=sorted(train_v), test_videos=sorted(test_v),
                          train_rows=len(train_ids), test_rows=len(test_ids)))
    rows = rows.drop(columns="features", errors="ignore").assign(probability=predictions, fold=assignments)
    rows.to_csv(OUT/"oof.csv", index=False)
    save_json(OUT/"FOLDS.json", folds)
    metrics = dict(rows=len(rows), videos=int(rows.video_id.nunique()), folds=FOLDS,
        positives=int(y.sum()), rate=float(y.mean()), auc=float(roc_auc_score(y, predictions)),
        log_loss=float(log_loss(y, predictions)), brier=float(brier_score_loss(y, predictions)),
        constant_log_loss=float(log_loss(y, constant)))
    save_json(OUT/"METRICS.json", metrics)
    scaler, model = fit(x, y)
    save_json(MODEL_PATH, dict(valid=True, production_enabled=False, columns=COLUMNS,
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]), training=metrics,
        label_source="observed_garbage_increase_interval", supported_receiver="未発火側のみ"))


def main() -> None:
    """分類完了と事前登録を確認してから学習する。"""
    assert Path("logs/e19/CLASSIFICATION.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT/"STATUS.json", dict(stage="FEATURES"))
    rows = prepare(inputs())
    save_json(OUT/"STATUS.json", dict(stage="CV"))
    train(rows)
    save_json(OUT/"STATUS.json", dict(stage="COMPLETE"))


if __name__ == "__main__":
    main()
