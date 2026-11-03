"""段2採点器を共有し、事前登録の門v2と旧条件を併記する。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts import gfe_phase_stage2_score_20261002 as previous
from scripts.gfe_phase_stage2b_train_20261002 import OUT
from scripts.gfe_phase_stage2_replay_20261002 import SOURCES, EXISTING
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from src.exchange_event_evaluator import file_sha256
import numpy as np

STAGE2 = Path('logs/gfe_phase_stage2_20261002')
MODES = ('baseline', 'V3s', 'R1s')
USED_SCORERS = ('prefire_v6_score.py', 'prefire_v6_regression.py', 'prefire_gate_report_20260930.py')
TIME_POINT, TIME_UPPER, AGREE_POINT, AGREE_LOWER = .002, .020, -.005, -.030


def root(mode: str, source: str) -> Path:
    """本番は段2保存値をそのまま再利用する。"""
    return (STAGE2 if mode == 'baseline' else OUT)/mode/'replay'/source


def usage_latency(mode: str) -> dict:
    """用途が発生した通知を母数に、静止・着弾を別々に集計する。"""
    result = {}
    for purpose in ('static', 'landing'):
        total, inference, calls = [], [], 0
        models = set()
        for source in SOURCES:
            with np.load(root(mode, source)/'gfe_usage.npz') as data:
                mask = data['purpose'] == purpose
                calls += int(mask.sum())
                models.update(data['model'][mask].tolist())
                total.extend(data[purpose+'_notification_ms'])
                inference.extend(data[purpose+'_inference_ms'])
        assert len(total) == len(inference) > 0
        result[purpose] = dict(notifications=len(total), calls=calls, models=sorted(models),
            p95_ms=float(np.percentile(total, 95)), inference_p95_ms=float(np.percentile(inference, 95)),
            maximum_ms=float(np.max(total)))
    return result


def gate_v2(sets: list, decisions: dict, metric: object) -> None:
    """条件1・4・6を保持し、2′・3′のみ合算115試合で追加する。"""
    for mode in MODES[1:]:
        arrays = [np.asarray([r['agree'] for s in sets for r in s['rows'][m]])
                  for m in ('baseline', mode)]
        agree = metric.paired(arrays[1], arrays[0], True)
        agree['candidate'] = agree.pop('phase6')
        result = decisions[mode]
        result['overall_agree'] = agree
        value = result['overall']
        gates = {k: v for k, v in result['gates'].items() if k not in ('condition2', 'condition3')}
        gates['condition2_prime'] = value['difference'] <= TIME_POINT and value['ci95'][1] < TIME_UPPER
        gates['condition3_prime'] = (agree['difference'] >= AGREE_POINT and agree['ci95'][0] > AGREE_LOWER
            and all(s['flips'][mode] <= previous.FLIP_RATIO*s['flips']['baseline'] for s in sets))
        result['gates_v2'], result['passed_v2'] = gates, all(gates.values())


def verify_inputs() -> dict:
    """OFF照合と全再生の固定構成、採点関数のハッシュを検査する。"""
    byte_identity = {}
    protocol = json.loads((STAGE2/'PROTOCOL.json').read_text())
    prior = json.loads((STAGE2/'METRICS.json').read_text())
    hashes = {name: file_sha256(previous.PREFIRE/'scripts'/name) for name in USED_SCORERS}
    assert hashes == {n: prior['scorer_sha256'][n] for n in USED_SCORERS}, '段2から採点器が変わっている'
    for source in EXISTING:
        directory = OUT/'baseline/replay'/source
        value = json.loads((directory/'BYTE_IDENTITY.json').read_text())
        assert all(value.values())
        for name in ('display.npz', 'events.jsonl'):
            assert (directory/name).read_bytes() == (root('baseline', source)/name).read_bytes()
        byte_identity[source] = value
    for mode in MODES[1:]:
        for source in SOURCES:
            done = json.loads((root(mode, source)/'DONE.json').read_text())
            assert done['state'] == 'completed' and done['options'] == protocol['options']
            assert all(p['num_threads'] == 1 for p in done['threadpools'] if p['user_api'] == 'blas')
    return dict(byte_identity=byte_identity, scorer_sha256=hashes, options=protocol['options'])


def main() -> None:
    """全表保存後も本番設定を変更せず、採否を統括へ渡す。"""
    os.nice(max(0, 19-os.nice(0)))
    memory_guard()
    previous.PREFIRE = OUT/'scorer_reference'
    verification = verify_inputs()
    previous.OUT, previous.MODES, previous.root = OUT, MODES, root
    metric = previous.metric_functions()
    sets = [previous.score_set(previous.EVALSET/'labels.json', previous.SET1, metric),
            previous.score_set(previous.EVALSET/'set2/labels.json', previous.SET2, metric)]
    regressions = {m: previous.regression(m, metric) for m in MODES}
    speeds = {m: previous.latency(m) for m in MODES}
    decisions = previous.judge(sets, regressions, speeds, metric)
    gate_v2(sets, decisions, metric)
    for mode in MODES[1:]:
        speeds[mode]['by_purpose'] = usage_latency(mode)
    result = dict(status='COMPLETE', set1=sets[0], set2=sets[1], regression=regressions,
        latency=speeds, decisions=decisions, adopted=False, verification=verification)
    save_json(OUT/'METRICS.json', result)
    from scripts.gfe_phase_stage2b_routes_20261002 import analyze
    result['routes'] = analyze(result)
    save_json(OUT/'METRICS.json', result)
    from scripts.gfe_phase_stage2b_report_20261002 import write_summary
    write_summary(result)
    print(json.dumps(decisions, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
