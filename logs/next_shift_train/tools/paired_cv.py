"""148動画CV (同じ行・fold・seed) の新旧対応比較と動画単位ブートストラップ95%CI。
使い方: python paired_cv.py <old_dir> <new_dir> <glob> <cols...>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
old, new, pattern, cols = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
key = ['seed', 'row_id']
a = pd.concat([pd.read_csv(p) for p in sorted(old.glob(pattern))])
b = pd.concat([pd.read_csv(p) for p in sorted(new.glob(pattern))])
m = a.merge(b, on=key, suffixes=('_old', '_new'), validate='one_to_one')
assert len(m) == len(a) == len(b)
y = m.label_old.to_numpy(); videos = m.video_id_old.to_numpy()
uniq = np.unique(videos); idx = {v: np.flatnonzero(videos == v) for v in uniq}
rng = np.random.default_rng(0)
out = dict(rows=len(m), videos=len(uniq))
for c in cols:
    po, pn = m[c + '_old'].to_numpy(), m[c + '_new'].to_numpy()
    res = dict(ll_old=log_loss(y, po), ll_new=log_loss(y, pn), auc_old=roc_auc_score(y, po), auc_new=roc_auc_score(y, pn))
    d_ll, d_auc = [], []
    for _ in range(300):
        s = np.concatenate([idx[v] for v in rng.choice(uniq, len(uniq))])
        d_ll.append(log_loss(y[s], pn[s], labels=[0, 1]) - log_loss(y[s], po[s], labels=[0, 1]))
        d_auc.append(roc_auc_score(y[s], pn[s]) - roc_auc_score(y[s], po[s]))
    res.update(d_ll=res['ll_new'] - res['ll_old'], d_ll_ci95=np.quantile(d_ll, [.025, .975]).tolist(),
               d_auc=res['auc_new'] - res['auc_old'], d_auc_ci95=np.quantile(d_auc, [.025, .975]).tolist())
    out[c] = res
print(json.dumps(out, indent=1))
