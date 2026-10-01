"""診断: v3.final_fit と同じ学習を random_state だけ変えて行い、再学習ゆらぎの大きさを測る (門の対象外)。
使い方: python refit_seed.py <run_dir(rows.csv,S1_prime.npy,S3_prime.npy)> <models_out> <random_state> <shared_directory>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scripts import train_exchange_event_models_v3_20260928 as v3
v2, v1 = v3.v2, v3.v1
run, models, seed, shared = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
models.mkdir(parents=True, exist_ok=True)
rows = pd.read_csv(run / 'rows.csv')
target = np.where(rows.sign > 0, rows.label, 1 - rows.label)
manifest = dict(models={}, production_enabled=False, version='E15_live_count', shared_directory=shared,
                ablation=f'診断: random_state={seed}, 特徴={run}')
for stage, columns in (('S1', v2.S1_COLUMNS), ('S3', v2.S3_COLUMNS)):
    x = np.load(run / (stage + '_prime.npy'), mmap_mode='r')
    for suffix in ('', '_light'):
        name = stage + '_prime' + suffix
        model = v2.HistGradientBoostingClassifier(random_state=seed, **(v1.LIGHT_PARAMS if suffix else {}))
        model.fit(x, target, sample_weight=rows.weight)
        path = models / (name + '.joblib')
        v2.joblib.dump(model, path)
        manifest['models'][name] = dict(file=path.name, sha256=v1.file_sha256(path),
            columns=columns + v2.NEW_COLUMNS, valid=True, version='E15_live_count')
v1.save_json(models / 'manifest.json', manifest)
print('ok')
