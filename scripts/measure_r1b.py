"""R1bは残差を参考値とし、遅延をOFF/R1/R1bの同じ置き集合で比較する。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts import measure_r1_recognition as measure, r1_measure_helpers as helpers
from scripts._d3_inventory import SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/r1b')


def common_delays(previous: list[dict], current: list[dict]) -> dict:
    """対応欠測を明示し、同じ置きだけから3条件の分位点を計算する。"""
    assert len(previous) == len(current)
    values = dict(off=[], r1=[], r1b=[])
    missing = dict(off=0, r1=0, r1b=0)
    indices = []
    for index, (before, after) in enumerate(zip(previous, current)):
        assert all(before[k] == after[k] for k in ('source', 't_sec', 'side'))
        assert before['old_delay'] == after['old_delay'], '接地基準またはOFF反映が変化'
        row = dict(off=before['old_delay'], r1=before['new_delay'], r1b=after['new_delay'])
        for key, value in row.items():
            missing[key] += value is None
        if any(value is None for value in row.values()):
            continue
        indices.append(index)
        for key, value in row.items():
            values[key].append(value)
    quantiles = {key: helpers.quantiles(value) for key, value in values.items()}
    same = bool(indices) and all(np.isclose(quantiles['off'][key], quantiles['r1b'][key],
        atol=measure.FRAME_TOLERANCE, rtol=0) for key in ('p50', 'p95'))
    return dict(total=len(current), paired=len(indices), indices=indices,
                missing=missing, quantiles=quantiles, passed=same)


def main() -> None:
    """D3の残差・正解率は固定母数を維持し、採否の遅延だけ共通母数に揃える。"""
    assert (OUT/'COLLECT_COMPLETE.json').exists()
    measure.OUT = helpers.OUT = OUT
    events = json.loads(Path('logs/d3/measurements.json').read_text())
    rows = [r for s in SOURCES for r in measure.measure_source(s, events)]
    result = measure.summarize(rows)
    previous = json.loads(Path('logs/r1/RECOGNITION_ROWS.json').read_text())
    delays = common_delays(previous, rows)
    result['reference_a'] = result['gates'].pop('a')
    result['gates']['c'] = delays['passed']
    result['common_delays'] = delays
    save_json(OUT/'RECOGNITION_ROWS.json', json.loads(json.dumps(rows, default=int)))
    save_json(OUT/'RECOGNITION_METRICS.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'common_delays'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
