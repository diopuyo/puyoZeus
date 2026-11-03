"""同じ先頭20分のABBA原票を、更新処理量と実公開遅延を分けて集計する。"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_live_obs_20261020d import analyze, feature_parts, phases
from summarize_live_obs_speed import combined, RATIO
from summarize_live_obs_time_gate import read, stats
from summarize_live_obs_actual_latency import summarize as actual_latency

ROOT = Path('D:/puyo_analyzer/packaging/abba_20261021')
DURATION_SEC = 1200
RUNS = ('A1_f_r2', 'B1_c_r2', 'B2_c_r2', 'A2_f')
LABELS = ('f', 'c')
CANDIDATE_LABEL = None


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def display_faults(run: Path, component: str = 'v49f') -> dict:
    """旧cは失敗票を持たないため、実際の公開DTOからも異常を判定する。"""
    failures = {}
    for row in read(run/'polls.jsonl'):
        snapshot = row['snapshot']
        value = ((snapshot.get('evaluations') or {}).get(component) or {})
        reason = value.get('reason') or ''
        if 'Error:' not in reason and '再起動' not in reason:
            continue
        if reason not in failures:
            failures[reason] = dict(first_poll_at=row['at'],
                first_source_sec=snapshot['timing']['source_available_ms']/1000,
                count=0)
        failures[reason]['count'] += 1
    return failures


def one_run(name: str) -> tuple[dict, list[dict]]:
    run = ROOT/name
    receipt = json.loads((run/'receipt.json').read_text(encoding='utf-8'))
    result = analyze(run)
    save(run/'analysis.json', result)
    rows = [r for r in combined(run/'output') if 0 <= r['t_sec'] < DURATION_SEC]
    parts = dict(m3=feature_parts(run/'output'))
    phase_rows = [phases(r, parts) for r in rows]
    raw = stats([sum(r['measured_ms'].values()) for r in rows])
    low = stats([r['total_low_spec_ms'] for r in rows])
    faults, ama_faults = display_faults(run), display_faults(run, 'ama_plain')
    actual = actual_latency(run)
    healthy = bool(receipt.get('all_three') and not faults and not ama_faults and not receipt.get('ama_worker_fatal')
                   and not receipt.get('evaluation_errors') and receipt['shutdown']['passed'])
    value = dict(run=name, receipt=receipt, updates=len(rows), v49f_faults=faults, ama_faults=ama_faults,
        healthy_three_frames=healthy,
        first_source_sec=min(r['t_sec'] for r in rows), last_source_sec=max(r['t_sec'] for r in rows),
        raw_ms=raw, legacy_low_spec_ms=low,
        actual_capture_to_http_ms=actual['capture_to_http_available_ms'],
        actual_b3_input_to_http_ms=actual['b3_input_to_http_available_ms'],
        legacy_over_500_rate=low['over_500']/len(rows),
        phase_raw_ms={k:stats([r[k] for r in phase_rows]) for k in phase_rows[0]},
        outlier_phase_counts=result['phase_summary'], signals=result['outlier_signals'],
        ama=result['ama'], ama_visible={k:v for k,v in result['ama_visible'].items() if k!='rows'},
        resources=result['evidence']['resources'], old_monitor_classification=result['measurement'],
        full_outliers=str(run/'analysis.json'))
    save(run/'summary_20min.json', value)
    return value, rows


def environment_samples(name: str, samples: list[dict]) -> list[dict]:
    """終了後の静穏待ちを前runのCPU統計へ混ぜない。原票はすべて保持する。"""
    selected = [s for s in samples if s.get('active_run') == name]
    receipt = ROOT/(name+'.exit.json')
    if receipt.exists():
        ended = datetime.fromisoformat(json.loads(receipt.read_text())['at'])
        selected = [s for s in selected if datetime.fromisoformat(s['at']) <= ended]
    return selected


def environment() -> dict:
    samples = read(ROOT/'environment_r2.jsonl')
    result = {}
    for name in RUNS:
        selected = environment_samples(name, samples)
        result[name] = dict(samples=len(selected),
            minimum_free_gib=min(s['free_gib'] for s in selected),
            minimum_c_free_gib=min(s['c_free_gib'] for s in selected),
            errors=[s for s in selected if 'error' in s],
            performance={str(cpu):stats([r['PercentProcessorPerformance'] for s in selected
                for r in s['clocks'] if r['Name']==f'0,{cpu}']) for cpu in range(os.cpu_count() or 1)},
            frequency_mhz={str(cpu):stats([r['ProcessorFrequency'] for s in selected
                for r in s['clocks'] if r['Name']==f'0,{cpu}']) for cpu in range(os.cpu_count() or 1)},
            resident_cpu_percent_one_core={name:stats([sum(r['PercentProcessorTime']
                for r in s['processes'] if r['Name'].split('#')[0] == name) for s in selected])
                for name in ('Discord', 'obs64', 'steam', 'msedge', 'codex')},
            vm_percent_one_core=stats([r['PercentProcessorTime'] for s in selected
                for r in s['processes'] if r['Name'].startswith('vmmem')]),
            battery_statuses=sorted({r['BatteryStatus'] for s in selected for r in s['battery']}))
    return result


def main() -> None:
    summaries, all_rows = {}, {}
    for name in RUNS:
        print('ANALYZE', name, flush=True)
        summaries[name], all_rows[name] = one_run(name)
    pooled = {}
    for version, names in ((LABELS[0], (RUNS[0], RUNS[3])), (LABELS[1], (RUNS[1], RUNS[2]))):
        values = [row for name in names for row in all_rows[name]]
        pooled[version] = dict(raw_ms=stats([sum(r['measured_ms'].values()) for r in values]),
            legacy_low_spec_ms=stats([r['total_low_spec_ms'] for r in values]),
            actual_capture_to_http_ms=stats([r['actual_ms'] for name in names
                for r in read_actual(ROOT/name)]),
            actual_b3_input_to_http_ms=stats([r['actual_ms'] for name in names
                for r in read_actual(ROOT/name, 'b3_rows')]))
    cutoff = min([DURATION_SEC]+[v['first_source_sec'] for s in summaries.values()
                                for kind in ('v49f_faults', 'ama_faults') for v in s[kind].values()])
    prefix = {name:stats([r['total_low_spec_ms'] for r in rows if r['t_sec'] < cutoff])
              for name, rows in all_rows.items()}
    v49f_cutoff = min([DURATION_SEC]+[v['first_source_sec'] for s in summaries.values()
                                    for v in s['v49f_faults'].values()])
    v49f_prefix = {name:stats([r['total_low_spec_ms'] for r in rows if r['t_sec'] < v49f_cutoff])
                  for name, rows in all_rows.items()}
    candidate = LABELS.index(CANDIDATE_LABEL or LABELS[0])
    candidate_runs = (RUNS[0], RUNS[3]) if candidate == 0 else (RUNS[1], RUNS[2])
    save(ROOT/'comparison.json', dict(duration_sec=DURATION_SEC, order=RUNS, ratio=RATIO,
        labels=LABELS, candidate=LABELS[candidate],
        runs=summaries, pooled=pooled, environment=environment(),
        healthy_three_frames_all=all(s['healthy_three_frames'] for s in summaries.values()),
        candidate_healthy_three_frames=all(summaries[n]['healthy_three_frames'] for n in candidate_runs),
        before_first_v49f_error=dict(cutoff_source_sec=v49f_cutoff, legacy_low_spec_ms=v49f_prefix),
        before_first_display_error=dict(cutoff_source_sec=cutoff, legacy_low_spec_ms=prefix,
            note='最初に例外が公開された動画時刻までの補助比較。長い停止区間の影響を減らすが、'
                 '例外発生から公開までの短い遅れは残る。先頭20分の値を置き換えない。'),
        note='同じ動画先頭20分。処理量の和は公開遅延そのものではない。旧換算係数と実msを別記。'
             'リアルタイム観測列は負荷で変わる。旧CPU0〜7での58件と件数を直接比較しない。'))
    print('DONE', flush=True)


def read_actual(run: Path, key: str = 'rows') -> list[dict]:
    return json.loads((run/'actual_latency.json').read_text(encoding='utf-8'))[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--runs', nargs=4, default=RUNS)
    parser.add_argument('--labels', nargs=2, default=LABELS)
    parser.add_argument('--candidate-label')
    options = parser.parse_args()
    ROOT, RUNS = options.root, tuple(options.runs)
    LABELS = tuple(options.labels)
    CANDIDATE_LABEL = options.candidate_label
    main()
