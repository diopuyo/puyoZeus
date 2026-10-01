"""再利用の条件確認: rs0 の S′/G_fe が既存の v3・v1 (旧) / v5・v5_common (補正) と予測一致するか、G_fe が random_state に依らないか。"""
import json
from pathlib import Path
import joblib, numpy as np
E15 = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15')
out = {}
for v, s_ref, c_ref, feat in (('orig', 'models/exchange_event_v3', 'models/exchange_event_v1', E15),
                              ('T', 'models/exchange_event_v5', 'models/exchange_event_v5_common', Path('logs/next_shift_train/sprime_T'))):
    for key in ('S1_prime_light', 'S3_prime_light', 'S1_prime', 'S3_prime'):
        x = np.load(feat / (key.split('_')[0] + '_prime.npy'))[:20000]
        a = joblib.load(f'{s_ref}/{key}.joblib').predict_proba(x)[:, 1]
        b = joblib.load(f'models/rejudge/{v}_rs0/{key}.joblib').predict_proba(x)[:, 1]
        out[f'{v}|{key}|rs0_vs_ref_maxdiff'] = float(np.abs(a - b).max())
    g = [joblib.load(p) for p in (f'{c_ref}/G_fe.joblib', f'models/rejudge/{v}_rs0_common/G_fe.joblib', f'models/rejudge/{v}_rs3_common/G_fe.joblib')]
    out[f'{v}|G_fe_rs0_vs_ref_maxcoef'] = float(np.abs(g[0].coef_ - g[1].coef_).max())
    out[f'{v}|G_fe_rs3_vs_rs0_maxcoef'] = float(np.abs(g[2].coef_ - g[1].coef_).max())
print(json.dumps(out, indent=1))
