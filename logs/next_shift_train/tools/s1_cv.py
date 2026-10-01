"""S1′(+light) を v3 fold_job と同じ fold・重み・seed で CV する (v3 の CV は S3′ だけなので補う)。
使い方: python s1_cv.py <run_dir> <out_dir>   run_dir に rows.csv と S1_prime.npy"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scripts import train_exchange_event_models_v3_20260928 as v3
v2, v1 = v3.v2, v3.v1
run, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = pd.read_csv(run / 'rows.csv'); x = np.load(run / 'S1_prime.npy')
target = np.where(rows.sign > 0, rows.label, 1 - rows.label)
for seed in v1.SEEDS:
    for fold in v1.FOLDS:
        path = out / f'seed_{seed}/fold_{fold}/predictions.csv'
        if path.exists():
            continue
        ref = pd.read_csv(v1.run_path(v2.REFERENCE, 'base') / f'seed_{seed}/fold_{fold}/de_predictions.csv')
        test = rows.video_id.isin(ref.video_id.unique()).to_numpy()
        result = rows.loc[test].copy()
        for name, params in (('S1_prime', {}), ('S1_prime_light', v1.LIGHT_PARAMS)):
            model = v2.HistGradientBoostingClassifier(random_state=v2.SEED, **params)
            model.fit(x[~test], target[~test], sample_weight=rows.weight.to_numpy()[~test])
            p = model.predict_proba(x[test])[:, 1]
            result[name] = np.where(rows.sign.to_numpy()[test] > 0, p, 1 - p)
        result['seed'], result['fold'] = seed, fold
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
print('done')
