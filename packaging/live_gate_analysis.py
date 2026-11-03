"""係数測定専用の全長集計。公開の入口時刻や並行工程の合計は使わない。"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_time_gate import stats

FIRST_SEC = 1200
FLOW_TAIL_TOLERANCE_SEC = 10
MIN_FLOW_FRAMES = 100
BLOCK_SEC = 1200
MS = 1000.
RUNS = ('A1_normal', 'B1_low', 'B2_low', 'A2_normal')


def records(path: Path) -> Iterator[dict]:
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            yield json.loads(line)


def validate_flow(run: Path, duration: float) -> dict:
    receipt = json.loads((run/'receipt.json').read_text(encoding='utf-8'))
    if receipt.get('exit_code') != 0 or receipt.get('error') or receipt.get('evaluation_errors') or (
            receipt.get('ama_worker_fatal')) or not receipt.get('shutdown', {}).get('passed'):
        raise ValueError(f'{run.name}: 正常終了・例外0を確認できない')
    frames, digests, scores, bins = set(), set(), set(), set()
    times, clocks = [], []
    for row in records(run/'polls.jsonl'):
        snapshot = row['snapshot']
        timing, values = snapshot.get('timing', {}), snapshot.get('evaluations', {})
        stamp = timing.get('source_available_ms', 0)/MS
        frames.add(timing.get('source_available_frame', 0))
        times.append(stamp)
        clocks.append(row['at'])
        b3 = values.get('practical') or {}
        if b3.get('input_digest'):
            digests.add(b3['input_digest'])
        if b3.get('p1_win_probability') is not None:
            scores.add(b3['p1_win_probability'])
        if all((values.get(key) or {}).get(field) is not None for key, field in
               (('practical', 'p1_win_probability'), ('v49f', 'p1'), ('ama_plain', 'p1'))):
            bins.add(int(stamp//BLOCK_SEC))
    expected = set(range(math.ceil(duration/BLOCK_SEC)))
    valid = (len(frames) >= MIN_FLOW_FRAMES and len(scores) > 1 and times and
             min(times) < FLOW_TAIL_TOLERANCE_SEC and max(times) >= duration-FLOW_TAIL_TOLERANCE_SEC
             and expected <= bins and clocks[-1]-clocks[0] >= duration-FLOW_TAIL_TOLERANCE_SEC)
    if not valid:
        raise ValueError(f'{run.name}: 全長の映像進行・3枠・実時間再生を確認できない')
    return dict(passed=True, distinct_frames=len(frames), distinct_input_digests=len(digests),
        distinct_scores=len(scores), first_sec=min(times), last_sec=max(times), three_panel_blocks=sorted(bins),
        wall_span_sec=clocks[-1]-clocks[0], definition='HTTPの映像フレーム進行と評価変化、各20分区間の3枠')


def delays(run: Path) -> dict[str, list[tuple[float, float]]]:
    path = run/'output/b3_time_gate.jsonl'
    sources = {}
    for row in records(path):
        if row.get('kind') == 'result' and row.get('accepted'):
            sources[row['game'], row['frame']] = row.get('captured_at'), row['t_sec']
    result = dict(latest_capture=[], b3_origin=[])
    seen_latest, seen_b3 = set(), set()
    for row in records(path):
        if row.get('kind') != 'publication':
            continue
        latest_key = row['frame'], row.get('captured_at')
        key = row['game'], row.get('state_frame')
        candidates = [('latest_capture', latest_key, seen_latest, (row.get('captured_at'), row['t_sec'])),
                      ('b3_origin', key, seen_b3, sources.get(key))]
        for metric, identity, seen, source in candidates:
            if source is None or identity in seen or source[0] is None:
                continue
            elapsed = (row['monotonic']-source[0])*MS
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('時計の不整合。負値を捨てて係数を作らない')
            seen.add(identity)
            result[metric].append((source[1], elapsed))
    return result


def summary(rows: list[tuple[float, float]], duration: float) -> dict:
    first = stats([value for stamp, value in rows if 0 <= stamp < FIRST_SEC])
    full = stats([value for stamp, value in rows if 0 <= stamp < duration])
    if not first['n'] or not full['n']:
        raise ValueError('実遅延の標本なし')
    blocks = [dict(start_sec=start, end_sec=min(start+BLOCK_SEC, duration),
        **stats([v for t, v in rows if start <= t < min(start+BLOCK_SEC, duration)]))
        for start in range(0, math.ceil(duration), BLOCK_SEC)]
    halves = [stats([v for t, v in rows if start <= t < end]) for start, end in
              ((0, duration/2), (duration/2, duration))]
    for group in (first, *blocks, *halves):
        group.pop('p99', None)
    first.pop('maximum', None)
    full.pop('p50', None)
    full.pop('p95', None)
    return dict(first20=first, full=full, blocks=blocks, halves=halves,
        half_p95_ratio=halves[1].get('p95', 0)/halves[0]['p95'] if halves[0].get('p95') else None,
        first20_over_500_fraction=first['over_500']/first['n'],
        full_over_500_fraction=full['over_500']/full['n'])


def ratios(low: dict, normal: dict) -> dict:
    result = {}
    for quantile, window in (('p50', 'first20'), ('p95', 'first20'), ('p99', 'full')):
        denominator = normal[window][quantile]
        if denominator <= 0:
            raise ValueError('通常条件の分母が正でない')
        result[quantile] = low[window][quantile]/denominator
    return result


def memory_and_affinity(run: Path, mask: int) -> dict:
    rows = list(records(run/'output/b3_resources.jsonl'))
    totals, peaks, priorities = [], {}, {}
    for row in rows:
        processes = [p for p in row['processes'] if p['name'].lower() != 'obs64.exe']
        for process in processes:
            if process.get('unavailable'):
                continue  # 採取中に終了した短命子は原票に残し、確認できたCPUだけを照合する。
            actual = process.get('affinity_mask')
            if actual is None or actual != mask:
                raise ValueError(f'{run.name}: 実プロセスのCPU制限を確認できない: {process}')
            pid = str(process['pid'])
            priorities.setdefault(pid, set()).add(process.get('priority_class'))
            peaks[pid] = max(peaks.get(pid, 0), process.get('peak_rss_bytes', 0))
        totals.append(sum(p.get('rss_bytes', 0) for p in processes))
    if not totals:
        raise ValueError('全長のメモリ原票なし')
    # 標本間隔は約5秒。経過時刻で前後半を分け、起動時の増加も残す。
    halfway = (rows[0]['monotonic']+rows[-1]['monotonic'])/2
    halves = [stats([total for row, total in zip(rows, totals) if
        (row['monotonic'] < halfway) == first])['p50'] for first in (True, False)]
    return dict(samples=len(rows), first_rss_bytes=totals[0], last_rss_bytes=totals[-1],
        peak_tree_rss_bytes=max(totals), half_median_rss_bytes=halves,
        half_growth_bytes=halves[1]-halves[0], per_pid_peak_rss_bytes=peaks,
        priority_classes={pid: sorted(values) for pid, values in priorities.items()},
        affinity_mask=mask, minimum_free_gib=min(r['free_gib'] for r in rows))


def environment(root: Path, info: dict) -> dict:
    result = {}
    for name in RUNS:
        end = json.loads((root/f'{name}.exit.json').read_text(encoding='utf-8'))['at']
        from datetime import datetime
        rows = [r for r in records(root/'environment.jsonl') if r.get('active_run') == name and
                datetime.fromisoformat(r['at']).timestamp() <= end]
        if not rows or any(r.get('error') for r in rows):
            raise ValueError(f'{name}: 環境記録が欠落')
        applied = json.loads((root/f'{name}.power_applied.json').read_text(encoding='utf-8'))
        plan = applied['plan']
        stamps = [datetime.fromisoformat(r['at']).timestamp() for r in rows]
        if stamps[0] > applied['at']+30 or end-stamps[-1] > 30 or any(
                b-a > 60 for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f'{name}: 環境記録に長い欠落がある')
        if any(plan not in r.get('active_power_plan', '').lower() or
               any(b['BatteryStatus'] == 1 for b in r['battery']) for r in rows):
            raise ValueError(f'{name}: 電源プランの変更またはバッテリー駆動を検出')
        mask = info['low_mask'] if name.endswith('low') else info['normal_mask']
        names = {f"0,{r['logical']}" for r in info['topology'] if mask & (1 << r['logical'])}
        clocks = [c for r in rows for c in r['clocks'] if c['Name'] in names]
        if not clocks or any(c.get('PercentProcessorPerformance') is None for c in clocks):
            raise ValueError(f'{name}: 対象CPUの周波数/Performanceの記録が欠落')
        perf = stats([c['PercentProcessorPerformance'] for c in clocks])
        # boost無効が機械側で無視された場合は、その値を正式係数にしない。
        if name.endswith('low') and perf['p95'] > 110:
            raise ValueError(f'{name}: boost無効後もPerformance P95>110%。条件を確認して再測定')
        totals = [sum(p.get('WorkingSetPrivate', 0) for p in r['processes']) for r in rows]
        result[name] = dict(samples=len(rows), cpu_frequency_mhz=stats([c['ProcessorFrequency'] for c in clocks]),
            processor_performance_percent=perf, minimum_free_gib=min(r['free_gib'] for r in rows),
            system_private_working_set_first=totals[0], system_private_working_set_last=totals[-1],
            system_private_working_set_peak=max(totals),
            application=memory_and_affinity(root/name, mask),
            raw_process_memory_and_load='environment.jsonl (全長、プロセス別)')
    return result


def analyze(root: Path, info: dict) -> dict:
    duration = info['video']['seconds']
    raw = {name: delays(root/name) for name in RUNS}
    summaries = {name: {metric: summary(rows, duration) for metric, rows in value.items()}
                 for name, value in raw.items()}
    pooled, coefficients = {}, {}
    for metric in ('b3_origin', 'latest_capture'):
        pooled[metric] = {condition: summary([r for name in RUNS if name.endswith(condition)
            for r in raw[name][metric]], duration) for condition in ('normal', 'low')}
        coefficient = ratios(pooled[metric]['low'], pooled[metric]['normal'])
        first = [v*coefficient['p95'] for name in RUNS if name.endswith('normal')
                 for t, v in raw[name][metric] if 0 <= t < FIRST_SEC]
        converted = stats(first)
        converted.pop('p99')
        converted.pop('maximum')
        coefficients[metric] = dict(pooled=coefficient, pairs=[
            ratios(summaries[b][metric], summaries[a][metric]) for a, b in
            (('A1_normal', 'B1_low'), ('A2_normal', 'B2_low'))],
            normal_first20_scaled_by_p95=converted,
            scaled_over_500_fraction=converted['over_500']/converted['n'],
            p95_under_500=converted['p95'] <= 500,
            over_500_threshold=None, overall_gate='割合の線は実測後にuser判断')
    return dict(runs=summaries, pooled=pooled, coefficients=coefficients, environment=environment(root, info),
        windows=dict(p50_p95='先頭20分、暖機除外なし', p99_max_memory='全長'),
        definition='公開後の監査monotonic−入力captured_at。同一入力の最初の公開。',
        primary='b3_origin', secondary='latest_capture（死亡確定などB3以外の公開も含む）',
        capture_clock='動画の予定取得時刻。復号/リサイズ/待ちを含む。HTTP取得/ブラウザ描画は含まない。')
