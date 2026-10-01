"""門③だけ先に計算する (再生待ちの間の区切り報告用)。"""
import json
from scripts.next_shift_rejudge_gate_20261001 import cv_diff, SEEDS
import numpy as np
cv = [cv_diff(k) for k in SEEDS]
print(json.dumps(dict(per_seed=[round(c['diff'], 6) for c in cv], rows=cv[0]['rows'], mean=float(np.mean([c['diff'] for c in cv])))))
