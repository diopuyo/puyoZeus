"""経路別の確率の鋭さ・識別・較正を同じ表示重みで測る。"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scripts._diag_gfe_stage2_analyze import OUT, display_frames, loss
from scripts.gfe_phase_stage1_20261002 import memory_guard

QUANTILES = (0, .01, .25, .5, .75, .99, 1)


def weighted_quantile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """重み付き経験分布の逆関数を使う。"""
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])/weights.sum()
    return values[order][np.minimum(np.searchsorted(cumulative, QUANTILES, side='left'), len(order)-1)]


def main() -> None:
    """行数重みと試合等重みの違いを明示して確率差も保存する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    display = display_frames()
    raw = pd.read_csv(OUT/'landing_display_detail.csv')
    raw['p0'], raw['p3'] = raw.gfe0, raw.gfe3
    metrics, differences = [], []
    for route, frame in [('G_fe', display[display.route == 'G_fe']),
                         ('S3_landing',display[display.route == 'S3_landing']), ('着弾内部G_fe',raw)]:
        y, w = frame.y.to_numpy(), frame.weight.to_numpy()
        for model,column in [('baseline','p0'),('V3','p3')]:
            p = frame[column].to_numpy()
            metrics.append(dict(route=route, model=model, frames=len(frame),
                LL=np.average(loss(p,y), weights=w), AUC=roc_auc_score(y,p,sample_weight=w),
                Brier=np.average((p-y)**2, weights=w), confidence=np.average(np.maximum(p,1-p),weights=w),
                mean_prediction=np.average(p,weights=w)))
        delta = (frame.p3-frame.p0).to_numpy()
        for weighting, quantiles in [('行数',np.quantile(delta,QUANTILES)),('試合等重み',weighted_quantile(delta,w))]:
            differences.append(dict(route=route, weighting=weighting,
                **dict(zip(map(str,QUANTILES),quantiles))))
    pd.DataFrame(metrics).to_csv(OUT/'probability_metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'probability_difference.csv',index=False)


if __name__ == '__main__':
    main()
