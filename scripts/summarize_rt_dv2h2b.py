"""D-v2h2b実動画の観測照合・開始判定・通知遅延・RSSを原票から集計する。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.falsedeath_replay_20261001 import enabled_flags
from scripts.summarize_rt_integration_20261002 import distribution, rows, MIB

ROOT = Path('logs/rt_dv2h2b_20261003')


def observation_summary(directory: Path, samples: list[dict]) -> dict:
    """周期標本に重複する証明を除き、未観測以外の照合と実入力を数える。"""
    evidence = {json.dumps(row, sort_keys=True): row for sample in samples
                for row in sample['post_counter_audit'] if 'completion_prefix_check' in row}
    for row in rows(directory, 'e35_check-*.jsonl'):
        evidence[json.dumps(row, sort_keys=True)] = row
    checks = [row['completion_prefix_check'] for row in evidence.values()]
    reasons = Counter(check['reason'] for check in checks)
    observations = rows(directory, 'e35_observation-*.jsonl')
    stages = [stage for row in observations for entry in row['entries']
              for stage in entry['stages'].values()]
    useful = [row for row in evidence.values()
              if row['completion_prefix_check']['reason'] != 'recent_unobserved']
    useful.sort(key=lambda row: len(row['completion_prefix_check'].get('observed_stages', {})),
                reverse=True)
    return dict(check_reasons=dict(reasons), unique_audit_checks=len(checks),
        formula_stage_samples=sum(s['formula'] is not None for s in stages),
        display_stage_samples=sum(s['display'] is not None for s in stages),
        all_changed_observations_received_result=all(r['result_received'] for r in observations),
        non_unobserved_examples=useful[:3],
        history_boundaries=rows(directory, 'h1_boundary-*.jsonl'))


def summarize() -> dict:
    """保存した全通知とRSS標本を使い、資源条件と起動構成も照合する。"""
    live = ROOT/'live'
    samples = rows(live/'audit', 'state-*.jsonl')
    notices = rows(live/'audit', 'notifications-*.jsonl')
    metrics = json.loads((live/'metrics.json').read_text())
    runtime = rows(live, 'runtime.jsonl')
    config = samples[-1]['configuration']
    flags = {flag: config[flag[2:].replace('-', '_')] for flag in enabled_flags('v2h2b')}
    gate = int((live/'run.log').read_text().splitlines()[0].split()[1])
    retention = {key: max(row['retained'].get(key, 0) for row in samples)
                 for key in {key for row in samples for key in row['retained']}}
    return dict(interval_sec=[2579, 2650], exit_code=int((live/'exit.txt').read_text()),
        enabled_flags=flags, all_enabled=all(flags.values()),
        worker_notification_ms=distribution([row['milliseconds'] for row in notices]),
        worker_rss_mib=distribution([row['rss_bytes']/MIB for row in samples]),
        worker_rss_first_last_mib=[samples[i]['rss_bytes']/MIB for i in (0, -1)],
        nice=sorted({row['nice'] for row in samples}), pools=samples[-1]['pools'],
        resource_gate_bytes=gate, retained_maxima=retention,
        frames=metrics['frames'], dropped_frames=metrics['dropped_frames'],
        evaluation_errors=max(r['evaluation_errors'] for r in runtime),
        evaluation_restarts=max(r['evaluation_restarts'] for r in runtime),
        processing_error_records=len(rows(live, 'evaluation_errors.jsonl')),
        observation=observation_summary(live/'audit', samples))


def main() -> None:
    """合否条件を記録し、未観測しか無い実走を配線成功としない。"""
    result = summarize()
    observation = result['observation']
    result['verified'] = (result['exit_code'] == 0 and result['all_enabled']
        and result['evaluation_errors'] == result['evaluation_restarts'] == 0
        and result['processing_error_records'] == 0 and result['nice'] == [10]
        and result['resource_gate_bytes'] >= 6*1024**3
        and all(v == '1' for v in result['pools'].values())
        and observation['all_changed_observations_received_result']
        and bool(observation['non_unobserved_examples'])
        and observation['formula_stage_samples'] > 0 and observation['display_stage_samples'] > 0)
    (ROOT/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
