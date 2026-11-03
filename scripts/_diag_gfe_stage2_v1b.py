"""同じ54列のLR参照を使い、入力構成とモデル種の差を補助診断する。"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT, STAGE2, display_frames, loss, grouped
from scripts.gfe_phase_stage1_20261002 import memory_guard


def main() -> None:
    """保存済みV1b表示だけを使い、追加学習せず経路別の差を測る。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    frame = display_frames()
    parts = []
    for source, rows in frame.groupby('source'):
        with np.load(STAGE2/'V1b/replay'/source/'display.npz') as saved:
            indices = np.searchsorted(saved['t_sec'], rows.t)
            np.testing.assert_array_equal(saved['t_sec'][indices], rows.t)
            np.testing.assert_array_equal(saved['source'][indices], rows.route)
            current = rows.copy()
            current['p1b'] = saved['display_p1'][indices]
            parts.append(current)
    frame = pd.concat(parts,ignore_index=True)
    summaries = []
    for name, old, new in [('V1b-baseline','p0','p1b'),('V3-V1b','p1b','p3')]:
        current = frame.copy()
        current['l0'],current['l3'] = loss(current[old],current.y),loss(current[new],current.y)
        current['delta'],current['dp'] = current.l3-current.l0,current[new]-current[old]
        result = grouped(current,['route'])
        result['comparison'] = name
        summaries.append(result)
    result = pd.concat(summaries,ignore_index=True)
    expected = json.loads((STAGE2/'METRICS.json').read_text())['decisions']['V1b']['overall']['difference']
    assert abs(result[result.comparison == 'V1b-baseline'].contribution.sum()-expected) < 1e-12
    result.to_csv(OUT/'variant_routes.csv',index=False)


if __name__ == '__main__':
    main()
