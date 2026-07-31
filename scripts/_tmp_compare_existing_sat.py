import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

feat = pd.read_csv('data/indicators_v2/saturation_subset_result.csv')
print(feat.shape)
print([c for c in feat.columns if 'saturat' in c or 'current_max' in c or 'build_ceil' in c])

def safe_auc(y, s):
    mask = ~np.isnan(s) & ~np.isnan(y)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return np.nan, int(mask.sum())
    auc = roc_auc_score(y[mask], s[mask])
    return max(auc, 1-auc), int(mask.sum())

won = feat[feat['won'].notna()].copy()
won['won'] = won['won'].astype(int)
tcr = won['tsumo_count_rate'].astype(float)
mid = won[(tcr>0.33)&(tcr<=0.67)]
print('全体 n=', len(won), '中盤 n=', len(mid))
for col in ['current_max_chain_raw','saturated_chain_count_raw','saturation_raw','saturation_margin']:
    if col not in won.columns:
        print(col, 'NOT FOUND'); continue
    y = mid['won'].values.astype(float)
    s = mid[col].values.astype(float)
    auc, n = safe_auc(y, s)
    print(f'[中盤] won~{col} AUC={auc:.4f} n={n}')
    y2 = won['won'].values.astype(float)
    s2 = won[col].values.astype(float)
    auc2, n2 = safe_auc(y2, s2)
    print(f'[全体] won~{col} AUC={auc2:.4f} n={n2}')
