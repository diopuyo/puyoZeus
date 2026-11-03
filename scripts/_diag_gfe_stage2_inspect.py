"""診断対象の保存形式を読み取り専用で確認する。"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd
from scripts.gfe_phase_stage1_20261002 import memory_guard, REFERENCE, datasets
from scripts.gfe_phase_stage2_replay_20261002 import record_path, OUT


def main() -> None:
    """最小限の行・列・記録例だけを表示する。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    rows, design, phases = datasets(REFERENCE)
    print('rows', rows.head().to_dict('records'), 'shape', design.shape, flush=True)
    print('phases', {k: (v.shape, v[:5].tolist()) for k,v in phases.items()}, flush=True)
    for source in ('c1', 'zenchi', 's0', 'q_7gc4TgFig'):
        with np.load(OUT/'baseline/replay'/source/'display.npz') as d:
            print(source, {k: (d[k].shape, d[k].flatten()[:4].tolist()) for k in d.files})
            print('sources', np.unique(d['source'], return_counts=True))
        seen, counts = set(), {}
        with gzip.open(record_path(source), 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                kind = r['kind']
                counts[kind] = counts.get(kind, 0)+1
                if kind not in seen and kind in ('header','static','static_call','display'):
                    print(source, kind, str(r)[:1800])
                    seen.add(kind)
        print('counts', counts, flush=True)
    print('predictions', pd.read_csv('logs/gfe_phase_stage1_20261002/predictions.csv', nrows=3).to_dict('records'))


if __name__ == '__main__':
    main()
