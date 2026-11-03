"""段2bのR1を全1,035,639行・固定HistGBC設定で最終学習する。"""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

from scripts import gfe_phase_stage1_20261002 as stage
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from src.exchange_event_evaluator import STATIC_GFE_VERSION, file_sha256
from src.exchange_event_features import G_COLUMNS, PHASE_BOUNDS, g_features

OUT = stage.ROOT / 'logs/gfe_phase_stage2b_20261002'
R1 = stage.ROOT / 'models/exchange_event_v1_r1_hgb'
EXPECTED_ROWS = 1035639


def model_directory(mode: str) -> Path:
    """既存V3は再利用し、R1のみ新規保存する。"""
    return R1 if mode == 'R1s' else stage.ROOT/'models/exchange_event_v1_userphase_v3'


def main() -> None:
    """M0外側OOF・局面閾値・向きを既存最終学習と揃える。"""
    os.nice(max(0, 19-os.nice(0)))
    resource.setrlimit(resource.RLIMIT_AS, (stage.MAX_MEMORY, stage.MAX_MEMORY))
    stage.memory_guard()
    started = time.monotonic()
    rows, design, phases = stage.datasets(stage.REFERENCE)
    assert len(rows) == EXPECTED_ROWS
    thresholds = np.quantile(phases['elapsed'], PHASE_BOUNDS)
    base = json.loads((stage.ROOT/'models/exchange_event_v1/manifest.json').read_text())
    np.testing.assert_array_equal(thresholds, base['elapsed_thresholds'])
    phase = np.column_stack((phases['fill'], np.searchsorted(thresholds, phases['elapsed'], side='left')))
    x = g_features(design, rows.A.to_numpy(), rows.sign.to_numpy(), phase)
    assert x.shape == (EXPECTED_ROWS, len(G_COLUMNS))
    model = HistGradientBoostingClassifier(**stage.HGB_PARAMS)
    model.fit(x, np.where(rows.sign.to_numpy() > 0, rows.label.to_numpy(), 1-rows.label.to_numpy()))
    stage.memory_guard()
    R1.mkdir(parents=True, exist_ok=False)
    path = R1/'G_fe.joblib'
    joblib.dump(model, path)
    entry = dict(file=path.name, sha256=file_sha256(path), columns=list(G_COLUMNS),
                 estimator_type=type(model).__name__, parameters=model.get_params())
    from threadpoolctl import threadpool_info
    meta = dict(version=STATIC_GFE_VERSION, variant='R1', models=dict(G_fe=entry),
        elapsed_thresholds=thresholds.tolist(), rows=len(rows), games=int(rows.game_id.nunique()),
        videos=int(rows.video_id.nunique()), m0='seed0 outer OOF: rows.A', adopted=False,
        optional_default=False, sklearn_version=sklearn.__version__, seconds=time.monotonic()-started,
        nice=os.nice(0), threadpools=threadpool_info())
    assert all(p['num_threads'] == 1 for p in meta['threadpools'])
    stage.save_json(R1/'manifest.json', meta)
    stage.save_json(OUT/'TRAIN_R1.json', meta)
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
