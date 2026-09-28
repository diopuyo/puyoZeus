"""B16の保存再生一致と、本番経路の短時間接続を検証する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scripts.measure_live_b6 import save
from src.exchange_event_record import read_records

ROOT = Path('logs/live_b16')
SMOKE_START, SMOKE_END = 3874.0, 3885.0


def verify(root: Path = ROOT) -> dict:
    reports = {key: json.loads((root/(key+'.json')).read_text()) for key in
               ('baseline_offline', 'fixed_offline', 'baseline_live', 'fixed_live', 'supervised_live')}
    original, fixed = (reports[k]['values'] for k in ('baseline_offline', 'fixed_offline'))
    assert len(original) == len(fixed)
    offline_diff = sum(a != b for a, b in zip(original, fixed))
    assert offline_diff == 0
    times = {r['t_sec'] for r in read_records(root/'recovered.jsonl.gz') if r['kind'] == 'display'}
    times.add(fixed[-1]['t_sec'])
    direct, remote = ({r['t_sec']: r for r in reports[k]['values'] if r['t_sec'] in times}
                       for k in ('fixed_live', 'supervised_live'))
    assert direct.keys() == remote.keys()
    remote_diff = sum(direct[t] != remote[t] for t in direct)
    assert remote_diff == 0
    assert reports['baseline_live']['failure']['kind'] == 'TypeError'
    assert reports['fixed_live']['values'][-1]['source'] == 'S3_landing'
    assert reports['fixed_live']['values'][-1]['p1'] is not None
    assert all(reports[k]['failure'] is None for k in ('fixed_live', 'fixed_offline', 'supervised_live'))
    result = dict(offline_notifications=len(fixed), offline_differences=offline_diff,
        supervised_publications=len(direct), supervised_differences=remote_diff,
        fixed_live_notifications=reports['fixed_live']['frames'],
        failure_sec=reports['baseline_live']['failure']['t_sec'],
        baseline_replay_seconds=reports['baseline_live']['seconds'],
        fixed_replay_seconds=reports['fixed_live']['seconds'])
    save(root/'verification.json', result)
    return result


def smoke(root: Path = ROOT, name: str = 'smoke') -> dict:
    from scripts.diagnose_live_b8 import case_command
    path = root/name
    path.mkdir(parents=True, exist_ok=False)
    args = case_command(('integration', False, SMOKE_START, SMOKE_END, 1, 0), path)
    config_path = Path(args[args.index('--config')+1])
    config = json.loads(config_path.read_text())
    config.update(runtime_audit=True, coalesce_features=True, event_priority=True, warmup_sec=10)
    save(config_path, config)
    with (path/'run.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((path/'integration/metrics.json').read_text())
    errors = path/'integration/evaluation_errors.jsonl'
    assert not errors.exists() or not errors.read_text().strip()
    assert report['frames'] > 0 and report['sse_probe_messages'] > 0
    result = dict(frames=report['frames'], sse_messages=report['sse_probe_messages'],
                  evaluation_errors=0, start=SMOKE_START, end=SMOKE_END)
    save(path/'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--name', default='smoke')
    options = parser.parse_args()
    print(json.dumps(smoke(name=options.name) if options.smoke else verify()))
