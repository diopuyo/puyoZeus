"""発火タイミング (hazard) モデルを148動画で学習する (Phase 2、2026-09-30)。

- 入力: logs/prefire_prediction/hazard_samples/video_*.npz (scripts/prefire_hazard_samples_20260930.py)
- 検証: 動画単位 GroupKFold(5) の OOF で AUC / log loss / Brier / ECE を出す (評価5記録は含まない)
- 「撃たないと窒息」行は規則で確率1とするため学習・評価から除き、その実発火率を別に数える (規則の検算)
- 出力: models/prefire_hazard_v1/{hazard.joblib, manifest.json}、logs/prefire_prediction/hazard_cv.json
使い方: PYTHONPATH=. python -m scripts.prefire_hazard_train_20260930
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.prefire_hazard_samples_20260930 import FIRE_MATCH_FRACTION, HORIZON_PLACEMENTS, OUT as SAMPLES
from src.prefire_exchange_sim import HAZARD_COLUMNS

MODEL_DIR = Path('models/prefire_hazard_v1')
REPORT = Path('logs/prefire_prediction/hazard_cv.json')
FOLDS = 5
ECE_BINS = 10
SEED = 20260930
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, random_state=SEED)


def load() -> dict[str, np.ndarray]:
    """全動画のサンプルを結合する (母数も返す)。"""
    parts = [dict(np.load(p)) | {'video': np.full(len(np.load(p)['label']), i)}
             for i, p in enumerate(sorted(SAMPLES.glob('video_*.npz')))]
    keys = ('features', 'label', 'label_missing', 'forced', 'video', 'best_send')
    data = {k: np.concatenate([p[k] for p in parts if len(p['label'])]) for k in keys}
    data['videos'] = np.asarray(len(parts))
    data['rows_total'] = np.asarray(sum(int(p['rows_total']) for p in parts))
    return data


def ece(y: np.ndarray, p: np.ndarray) -> float:
    """期待較正誤差 (等幅ビン)。"""
    bins = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    return float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                     for b in range(ECE_BINS) if (bins == b).any()))


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    """OOF の指標。"""
    return dict(auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p)),
                brier=float(brier_score_loss(y, p)), ece=ece(y, p))


def cross_validate(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """動画単位の OOF を HGB・ロジスティック・定数で比べる。"""
    oof = {name: np.zeros(len(y)) for name in ('hgb', 'logistic', 'constant')}
    for train, test in GroupKFold(FOLDS).split(x, y, groups):
        oof['hgb'][test] = HistGradientBoostingClassifier(**HGB_PARAMS).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        logistic = LogisticRegression(max_iter=2000).fit(x[train], y[train])
        oof['logistic'][test] = logistic.predict_proba(x[test])[:, 1]
        oof['constant'][test] = y[train].mean()
    return {name: scores(y, np.clip(p, 1e-7, 1 - 1e-7)) for name, p in oof.items()}


def save_model(x: np.ndarray, y: np.ndarray, report: dict) -> None:
    """全データで HGB を学習して保存し、manifest に sha256・列順・根拠を記録する。"""
    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / 'hazard.joblib'
    joblib.dump(HistGradientBoostingClassifier(**HGB_PARAMS).fit(x, y), path)
    manifest = dict(version='prefire_hazard_v1', file=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), columns=list(HAZARD_COLUMNS),
                    horizon_placements=HORIZON_PLACEMENTS, fire_match_fraction=FIRE_MATCH_FRACTION,
                    params=HGB_PARAMS, training='148動画 (boards_lean_phase_l_2026-08-11)、撃たないと窒息の行は除外',
                    cv=report['cv'], production_enabled=False)
    (MODEL_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')


def main() -> None:
    data = load()
    known = np.isfinite(data['label'])
    forced = data['forced'].astype(bool)
    use = known & ~forced
    x, y, groups = data['features'][use], data['label'][use].astype(int), data['video'][use]
    report = dict(videos=int(data['videos']), rows_total=int(data['rows_total']), samples=int(len(known)),
                  label_missing=int((~known).sum()), forced_rows=int((forced & known).sum()),
                  forced_fire_rate=float(data['label'][forced & known].mean()) if (forced & known).any() else None,
                  used=int(use.sum()), positive_rate=float(y.mean()), horizon_placements=HORIZON_PLACEMENTS)
    report['cv'] = cross_validate(x, y, groups)
    save_model(x, y, report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
