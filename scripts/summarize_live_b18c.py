"""B18cの全件一致と同条件コストを集約し、検収値を固定する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from scripts.profile_live_b18c import OUT, BASELINE_COMMIT
from scripts.verify_live_b18a import quantiles
from scripts.verify_live_b18b import save

VIDEO_ROOT = OUT/'final/video'
RESULT = Path('docs/B18C_RESULTS_20260929.json')
TARGET_P95_MS = 1.5
XML_REPORTS = ('tests.xml', 'final-unit.xml', 'limits.xml')
EXPECTED_TOTALS = (31346, 33146, 2091, 123, 57)


def test_counts() -> dict:
    cases = {}
    for name in XML_REPORTS:
        for case in ET.parse(OUT/name).iter('testcase'):
            cases[(case.get('classname'), case.get('name'))] = case
    skipped = sum(row.find('skipped') is not None for row in cases.values())
    failed = sum(row.find('failure') is not None or row.find('error') is not None for row in cases.values())
    return dict(passed=len(cases)-skipped-failed, skipped=skipped, failed=failed,
                reports=[str(OUT/name) for name in XML_REPORTS])


def source_report(source: str) -> dict:
    dest = VIDEO_ROOT/source
    report = json.loads((dest/'report.json').read_text())
    for key in ('windows', 'acceptance'):
        report[key].pop('mismatches', None)
    first = report['b18b_inputs']['first_mismatch']
    if first is not None:
        report['b18b_inputs']['first_mismatch'] = dict(index=first['index'])
    report.pop('cost')
    report['saved_input'] = json.loads((OUT/'saved'/source/'comparison.json').read_text())
    report['directory'] = str(dest)
    report['input_sha256'] = hashlib.sha256((dest/'inputs.jsonl.gz').read_bytes()).hexdigest()
    return report


def costs() -> dict:
    values: dict[str, list] = {}
    for source in ('review', 'zenchi'):
        with np.load(VIDEO_ROOT/source/'timing.npz') as timing:
            for key in timing.files:
                if key != 'before_total_sec':
                    values.setdefault(key, []).append(timing[key])
    combined = {key: np.concatenate(rows) for key, rows in values.items()}
    combined['before_retention_sec'] = combined['before_snapshot_sec']-combined['before_hsv_sec']
    return {key: quantiles(rows) for key, rows in combined.items()}


def validate_historical_inputs(row: dict) -> None:
    """旧review原票だけはB18b内の保持件数修正前。固定コミットとの比較を必須にする。"""
    inputs = row['b18b_inputs']
    if row['source'] != 'review':
        assert inputs['equal'] == inputs['total']
        return
    prefix = '.args.tuple.0.namespace.'
    expected = {prefix+'p2.namespace.prefire_snapshot.'+key: 97 for key in ('frames', 'start_sec')}
    expected.update({prefix+'p1.namespace.prefire_snapshot.'+key: 20
                     for key in ('board', 'end_sec', 'frames', 'reason', 'start_sec')})
    assert inputs['total'] == 6848 and inputs['equal'] == 6731
    assert inputs['different_fields'] == expected, inputs['different_fields']


def validate(result: dict) -> None:
    saved, updates, published, decisions, boards = 0, 0, 0, 0, 0
    for row in result['sources'].values():
        validate_historical_inputs(row)
        for key in ('b18b_probabilities', 'paired_observation'):
            assert row[key]['equal'] == row[key]['total'], (row['source'], key)
        for trace in (row['same_input_replay']['probabilities'], row['saved_input']['probabilities']):
            assert trace['equal'] == trace['total']
        for display in (row['b18b_display'], row['same_input_replay']['display'], row['saved_input']['display']):
            assert display['mismatched_frames'] == 0
        windows, adoption = row['windows'], row['acceptance']
        assert windows['boards'] == windows['reasons'] == windows['total']
        assert adoption['decisions'] == adoption['total'] and adoption['missing'] == adoption['extra'] == 0
        assert adoption['new_total'] == adoption['total']
        assert adoption['accepted_boards'] == adoption['hidden_distributions'] == adoption['accepted_total']
        saved += row['saved_input']['display']['compared_frames']
        updates += row['same_input_replay']['probabilities']['total']
        published += row['same_input_replay']['display']['compared_frames']
        decisions += adoption['total']
        boards += adoption['accepted_total']
    assert (saved, updates, published, decisions, boards) == EXPECTED_TOTALS
    assert result['tests']['failed'] == 0
    assert result['cost']['snapshot_sec']['P95_ms'] <= TARGET_P95_MS, result['cost']['snapshot_sec']


def main() -> None:
    result = dict(baseline_commit=BASELINE_COMMIT, nice=10, paired_alternating_order=True,
        sources={source: source_report(source) for source in ('review', 'zenchi')},
        cost=costs(), tests=test_counts(), target_p95_ms=TARGET_P95_MS,
        production_config_sha256=hashlib.sha256(Path('src/production_config.py').read_bytes()).hexdigest())
    save(RESULT, result)
    validate(result)
    print(json.dumps(dict(before=result['cost']['before_snapshot_sec'], after=result['cost']['snapshot_sec'],
                          tests=result['tests']), ensure_ascii=False))


if __name__ == '__main__':
    main()
