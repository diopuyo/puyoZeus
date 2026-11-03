"""全超過行の工程・CPU待ち上限・GC重なりを保存する（親子時間は合算しない）。"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_speed import combined, RATIO, ama_delay
from summarize_live_obs_time_gate import read, stats, resource_summary


def measurement_status(run: Path) -> dict:
    receipt = json.loads((run/'receipt.json').read_text())
    checks = read(run/'idle_checks.jsonl')
    external = read(run/'external_load.jsonl')
    active, previous = [], {}
    for sample in external:
        for row in sample['processes']:
            if row['name'].lower() == 'obs64.exe':
                continue
            if row['pid'] in previous and row.get('cpu_sec', 0)-previous[row['pid']] > .1:
                active.append(dict(pid=row['pid'], name=row['name'], at=sample['monotonic']))
        previous = {r['pid']: r.get('cpu_sec', 0) for r in sample['processes']}
    clocks = list(run.glob('cpu_clock_*.json'))
    candidate = bool(receipt['benchmark'] and checks and checks[-1]['busy'] == 0 and not active and clocks)
    return dict(official_candidate=candidate, benchmark_requested=receipt['benchmark'],
        active_external_samples=active, idle_last=checks[-1] if checks else None,
        clock_files=[str(p) for p in clocks],
        note='正式候補は開始前と5秒間隔標本の範囲で他の重い計算なし。OBSは許容。')


def evidence(root: Path, roles: dict, jobs: dict) -> dict:
    rows = [row for grouped in roles.values() for group in grouped.values() for row in group]
    keys = ('time_gate_write', 'time_gate_lock_wait', 'offer_lock_wait', 'publication_lock_wait',
            'spool_write', 'spool_truncate', 'spool_checkpoint_wait', 'spool_lock_wait',
            'fifo_append', 'fifo_popleft', 'fifo_clear', 'placement_audit_write', 'event_record_write',
            'journal_open', 'journal_flush', 'journal_close', 'journal_unlink',
            'event_ipc_wait', 'event_near_future_search')
    return dict(stages_ms={key: stats([r['stages'][key]['wall_ms'] for r in rows if key in r['stages']])
                          for key in keys},
        gc_ms={role: stats([r['gc_process_overlap_ms'] for group in values.values() for r in group
                           if 'gc_process_overlap_ms' in r])
               for role, values in roles.items()},
        ama_ms={key: stats([r[key] for group in jobs.values() for r in group if key in r])
                for key in ('queue_ms', 'compute_ms', 'cpu_ms', 'state_queue_ms', 'input_ms',
                            'slot_wait_ms', 'worker_delivery_ms')},
        ama_handoff=handoff(root),
        resources=resource_summary(read(root/'b3_resources.jsonl'), next(
            (r['pid'] for r in read(root/'ama_display.jsonl') if r.get('kind') == 'worker'), None)))


def handoff(root: Path) -> dict:
    rows = [r for r in read(root/'ama_display.jsonl') if r.get('received_at') is not None
            and r.get('completed_input_sec') == r.get('input_sec')]
    return {name: stats([(r[end]-r[start])*1000 for r in rows if r.get(start) is not None])
            for name, start, end in (
                ('worker_to_parent_ms', 'collected_at', 'received_at'),
                ('parent_to_evaluation_ms', 'received_at', 'displayed_at'))}


def visible_ama(root: Path, polls: list[dict]) -> dict:
    jobs = {(r['game'], r['result']['input_sec']): r for r in read(root/'ama_updates.jsonl')
            if 'value' in r.get('result', {})}
    display = {(r['game'][1], r['input_sec']): r for r in read(root/'ama_display.jsonl')
               if r.get('game') and r.get('input_sec') == r.get('completed_input_sec')}
    names = ('input_queue', 'input_processing', 'compute_queue', 'compute', 'collect',
             'ipc', 'evaluation', 'publication_and_poll')
    seen, rows = set(), []
    for poll in polls:
        value = poll['snapshot'].get('evaluations', {}).get('ama_plain', {})
        if value.get('p1') is None:
            continue
        candidates = [(key, job) for key, job in jobs.items() if key[1] == value.get('input_sec')
                      and job['requested_at'] <= poll['at']]
        if not candidates:
            continue
        key, job = max(candidates, key=lambda pair: pair[1]['requested_at'])
        if key in seen or key not in display:
            continue
        shown = display[key]
        times = [job.get(k) for k in ('requested_at', 'state_received_at', 'state_completed_at',
                 'began_at', 'completed_at', 'collected_at')]+[shown.get('received_at'),
                 shown['displayed_at'], poll['at']]
        if any(t is None for t in times) or any(b < a for a, b in zip(times, times[1:])):
            continue
        seen.add(key)
        costs = {name: (b-a)*1000 for name, a, b in zip(names, times, times[1:])}
        rows.append(dict(game=key[0], input_sec=key[1], total_ms=(times[-1]-times[0])*1000, phases_ms=costs))
    return dict(n=len(rows), phases_ms={name: stats([r['phases_ms'][name] for r in rows]) for name in names},
        rows=rows, note='同じ新規表示の依頼から最初のHTTP標本まで。行ごとの工程和は総遅延に一致。')


def ama_job(job: dict) -> dict:
    value = dict(queue_ms=(job['began_at']-job['requested_at'])*1000,
        compute_ms=(job['completed_at']-job['began_at'])*1000,
        cpu_ms=job['result'].get('cpu_sec', 0)*1000)
    for name, start, end in (
        ('state_queue_ms', 'requested_at', 'state_received_at'),
        ('input_ms', 'state_received_at', 'state_completed_at'),
        ('slot_wait_ms', 'state_completed_at', 'began_at'),
        ('worker_delivery_ms', 'completed_at', 'collected_at')):
        if start in job and end in job:
            value[name] = (job[end]-job[start])*1000
    return value


def overlaps(details: dict, jobs: list[dict]) -> dict:
    return {role: [dict(job_frame=job['frame'], overlap_ms=1000*max(0.,
                min(row['finished_at'], job['completed_at'])-max(row['started_at'], job['began_at'])))
            for row in details[role] for job in jobs
            if row['started_at'] < job['completed_at'] and job['began_at'] < row['finished_at']]
            for role in ('recognition', 'evaluation', 'publication')}


def timings(root: Path, role: str) -> dict:
    grouped = defaultdict(list)
    for row in read(root/f'spool/frame_timing_{role}.jsonl'):
        grouped[row['frame']].append(row)
    return grouped


def feature_parts(root: Path) -> dict:
    grouped = defaultdict(lambda: defaultdict(float))
    for row in read(root/'b3_state_timing.jsonl'):
        grouped[row['frame']]['state_ledger'] += row.get('state_ms', 0)+row.get('ledger_ms', 0)
        grouped[row['frame']]['state_ledger_cpu'] += row.get('state_cpu_ms', 0)+row.get('ledger_cpu_ms', 0)
    for row in read(root/'b3_updates.jsonl'):
        for key in ('search_ms', 'feature_only_ms', 'inference_ms',
                    'search_cpu_ms', 'feature_only_cpu_ms', 'inference_cpu_ms'):
            grouped[row['frame']][key] += row.get(key, 0)
    return grouped


def phases(row: dict, parts: dict) -> dict:
    """既存の処理量合計を分解。並列amaと待ち上限は別欄に残す。"""
    values = parts['m3'][row['frame']]
    features = sum(values.get(k, 0) for k in ('search_ms', 'feature_only_ms', 'state_ledger'))
    return dict(recognition=row['measured_ms']['recognition'], b3_m3_features=features,
        b3_m3_inference=values.get('inference_ms', 0),
        v49f=row['measured_ms']['v49f']+row['measured_ms']['v49f_baseline_v6'],
        display_dispatch=row['measured_ms']['display_dispatch'])


def phase_summary(outliers: list[dict]) -> dict:
    names = ('recognition', 'b3_m3_features', 'b3_m3_inference', 'v49f', 'display_dispatch')
    return {name: dict(total_ms=sum(r['phases_ms'][name] for r in outliers),
        over_500_alone=sum(r['phases_ms'][name]*RATIO > 500 for r in outliers),
        largest_count=sum(max(r['phases_ms'], key=r['phases_ms'].get) == name for r in outliers))
        for name in names}


def outlier_signals(outliers: list[dict]) -> dict:
    signals = Counter()
    for outlier in outliers:
        rows = [r for group in outlier['roles'].values() for r in group]
        signals['gc_instrumented'] += any('gc_process_overlap_ms' in r for r in rows)
        signals['gc_over_100ms'] += any(r.get('gc_process_overlap_ms', 0) > 100 for r in rows)
        for name, stages in (
            ('audit_write', ('time_gate_write', 'spool_write', 'spool_truncate', 'spool_checkpoint_wait',
                             'placement_audit_write', 'event_record_write')),
            ('fifo_io', ('fifo_append', 'fifo_popleft', 'fifo_clear')),
            ('journal_io', ('journal_open', 'journal_flush', 'journal_close', 'journal_unlink')),
            ('lock_wait', ('time_gate_lock_wait', 'spool_lock_wait', 'offer_lock_wait', 'publication_lock_wait'))):
            signals[name+'_instrumented'] += any(k in r['stages'] for r in rows for k in stages)
            signals[name+'_over_100ms'] += any(r['stages'].get(k, {}).get('wall_ms', 0) > 100
                                              for r in rows for k in stages)
        signals['evaluation_non_cpu_upper_over_100ms'] += outlier['evaluation_non_cpu_upper_ms'] > 100
    return dict(signals)


def public_status(run: Path) -> dict:
    receipt = json.loads((run/'receipt.json').read_text())
    reasons = Counter(r['snapshot'].get('evaluations', {}).get('ama_plain', {}).get('reason', '')
                      for r in read(run/'polls.jsonl'))
    failure = run/'output/ama_failure.json'
    return dict(receipt=receipt, ama_reasons=dict(reasons),
        ama_failure=json.loads(failure.read_text(encoding='utf-8')) if failure.exists() else None)


def analyze(run: Path) -> dict:
    root = run/'output'
    rows = combined(root)
    roles = {role: timings(root, role) for role in ('evaluation', 'publication', 'recognition', 'event_evaluation')}
    parts = {role: feature_parts(path) for role, path in (('m3', root), ('v6', root/'v49f_b3_v6'))}
    jobs = defaultdict(list)
    ama_jobs = [job for job in read(root/'ama_updates.jsonl') if 'value' in job.get('result', {})]
    for job in ama_jobs:
        jobs[job['frame']].append(ama_job(job))
    outliers = []
    for row in rows:
        if row['total_low_spec_ms'] <= 500:
            continue
        frame = row['frame']
        details = {role: values.get(frame, []) for role, values in roles.items()}
        outliers.append(dict(row, dominant=max(row['measured_ms'], key=row['measured_ms'].get),
            phases_ms=phases(row, parts),
            b3_features_ms={role: values[frame] for role, values in parts.items()},
            ama_parallel_not_added_ms=jobs[frame], roles=details,
            ama_compute_overlap_ms=overlaps(details, ama_jobs),
            evaluation_non_cpu_upper_ms=sum(max(0, r['wall_ms']-r['cpu_ms']) for r in details['evaluation'])))
    polls = read(run/'polls.jsonl')
    return dict(run=str(run), frames=len(rows), ratio=RATIO, measurement=measurement_status(run),
        polling_interval_ms=stats([(b['at']-a['at'])*1000 for a, b in zip(polls, polls[1:])]),
        total_low_spec_ms=stats([r['total_low_spec_ms'] for r in rows]),
        ama=ama_delay(root, read(run/'polls.jsonl')), evidence=evidence(root, roles, jobs),
        ama_visible=visible_ama(root, read(run/'polls.jsonl')),
        over_500=len(outliers), dominant_counts=dict(Counter(r['dominant'] for r in outliers)),
        phase_summary=phase_summary(outliers), outlier_signals=outlier_signals(outliers), public_status=public_status(run),
        ama_overlap_outliers=sum(any(r['ama_compute_overlap_ms'].values()) for r in outliers),
        outliers=outliers, note='GCはprocess内の重なり。wall-CPUはI/O・ロック・OS/GIL待ちの上限。'
        '旧原票では内訳を事後確定できない。amaは並列のためメイン合計に加算しない。')


if __name__ == '__main__':
    result = analyze(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'outliers'}, ensure_ascii=False))
