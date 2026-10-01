"""台の確認: 元参照で fit_g_fold(seed0, fold0) を再計算し、exev logs/exchange_event_v1 の g.csv と比べる。"""
import json
import numpy as np, pandas as pd
from scripts import train_exchange_event_models_20260926 as v1
new = v1.fit_g_fold(v1.REFERENCE, 0, 0)
old = pd.read_csv('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/exchange_event_v1/seed_0/fold_0/g.csv')
print(json.dumps(dict(rows=len(new), rows_old=len(old), row_ids_equal=bool(np.array_equal(new.row_id, old.row_id)),
    max_abs_diff=float(np.max(np.abs(new.G_fe.to_numpy() - old.G_fe.to_numpy()))))))
