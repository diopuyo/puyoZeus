"""再学習用の S1′/S3′ 行列と行表を元 (exev logs/e15) と比べ、変わった列・行を数える。"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scripts import train_exchange_event_models_v2_20260927 as v2
ref = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15'); new = Path(sys.argv[1])
a, b = pd.read_csv(ref/'rows.csv'), pd.read_csv(new/'rows.csv')
out = dict(rows_equal=bool(a.equals(b)), rows=len(a))
for name, cols in (('S1_prime', list(v2.S1_COLUMNS)+list(v2.NEW_COLUMNS)), ('S3_prime', None)):
    x, y = np.load(ref/f'{name}.npy'), np.load(new/f'{name}.npy')
    diff = ~np.isclose(x, y, equal_nan=True)
    names = cols if cols else [f'c{i}' for i in range(x.shape[1])]
    out[name] = dict(shape=list(x.shape), rows_changed=int(diff.any(1).sum()),
                     columns_changed={names[i]: int(diff[:, i].sum()) for i in np.flatnonzero(diff.any(0))})
print(json.dumps(out, ensure_ascii=False, indent=1))
