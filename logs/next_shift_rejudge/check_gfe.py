"""補正側 G_fe: v5_common と rejudge T_rs0 の差の大きさ (予測) と、スレッド数で再現するかを確認する。"""
import json, os, sys
import joblib, numpy as np
from scripts import train_exchange_event_models_20260926 as e1
from src.exchange_event_features import PHASE_BOUNDS, g_features
from pathlib import Path
rows, design, phases = e1.datasets(Path('logs/next_shift_train/gfe_T/reference'))
th = np.quantile(phases['elapsed'], PHASE_BOUNDS)
phase = np.column_stack((phases['fill'], np.searchsorted(th, phases['elapsed'], side='left')))
x = g_features(design, rows.A.to_numpy(), rows.sign.to_numpy(), phase)
a = joblib.load('models/exchange_event_v5_common/G_fe.joblib'); b = joblib.load('models/rejudge/T_rs0_common/G_fe.joblib')
pa, pb = a.predict_proba(x)[:, 1], b.predict_proba(x)[:, 1]
print(json.dumps(dict(threads=os.environ.get('OMP_NUM_THREADS'), n_iter=[int(a.n_iter_[0]), int(b.n_iter_[0])],
    pred_maxdiff=float(np.abs(pa - pb).max()), pred_meandiff=float(np.abs(pa - pb).mean()))))
