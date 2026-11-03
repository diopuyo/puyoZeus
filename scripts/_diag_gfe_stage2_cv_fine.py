"""表示時刻分布を1秒・相対進行1%単位で合わせる感度診断。"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT, OUTPUT, REFERENCE, datasets, display_frames, loss
from scripts.gfe_phase_stage1_20261002 import memory_guard

PROGRESS_BINS = 100


def main() -> None:
    """粗い時刻帯だけで結果を決めず、細かい再重み付けでも確かめる。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    rows, _, phases = datasets(REFERENCE)
    cv = pd.read_csv(OUTPUT/'predictions.csv', usecols=['row_id','label','V0','V3'])
    ids, d = cv.row_id.to_numpy(), display_frames()
    duration = rows.groupby('game_id').t_sec.transform('max').to_numpy()-phases['start']
    progress = np.divide(phases['elapsed'], duration, out=np.zeros(len(rows)), where=duration > 0)
    train_bands = {'seconds': np.floor(phases['elapsed']).astype(int),
                   'percent': np.minimum((progress*PROGRESS_BINS).astype(int), PROGRESS_BINS-1)}
    l0, l3 = loss(cv.V0.to_numpy(), cv.label.to_numpy()), loss(cv.V3.to_numpy(), cv.label.to_numpy())
    results = []
    for name, target in [('全表示',d), ('G_fe表示',d[d.route == 'G_fe']), ('S3_landing表示',d[d.route == 'S3_landing'])]:
        target_bands = {'seconds': np.floor(target.elapsed).astype(int),
                        'percent': np.minimum((target.progress*PROGRESS_BINS).astype(int), PROGRESS_BINS-1)}
        for axis, band in train_bands.items():
            n = max(band.max(), target_bands[axis].max())+1
            count = np.bincount(band[ids], minlength=n)
            mass = np.bincount(target_bands[axis], weights=target.weight, minlength=n)
            mass /= mass.sum()
            w = np.divide(mass, count, out=np.zeros(n), where=count > 0)[band[ids]]
            results.append(dict(target=name, resolution=axis, unsupported_mass=mass[count == 0].sum(),
                baseline=np.average(l0, weights=w), V3=np.average(l3, weights=w),
                difference=np.average(l3-l0, weights=w)))
    pd.DataFrame(results).to_csv(OUT/'cv_fine_reweighted.csv', index=False)


if __name__ == '__main__':
    main()
