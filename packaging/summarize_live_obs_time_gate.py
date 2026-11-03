"""本人OBS試験の§8原票を集計する。30分未満を合格と扱わない。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from io import StringIO
import math
from pathlib import Path
import statistics


def read(path: Path) -> list[dict]:
    # 完了原票を一括で読み、SMBへの細かなreadを繰り返さない。解析規則は同じ。
    return [json.loads(line) for line in StringIO(path.read_text(encoding='utf-8-sig'))] if path.exists() else []


def publication_rows(rows: list[dict]) -> list[dict]:
    """公開処理の入口ではなく、hub公開後の記録時刻で遅延を測る。"""
    return [dict(row, dispatch_started_at=row['published_at'], published_at=row['monotonic'])
            for row in rows if row['kind'] == 'publication']


def stats(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return dict(n=0)
    def quantile(q: float) -> float:
        index = (len(values)-1)*q
        lo = int(index)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(index-lo)
    return dict(n=len(values), p50=quantile(.5), p95=quantile(.95), p99=quantile(.99),
                maximum=values[-1], over_500=sum(v > 500 for v in values))


def time_rates(publications: list[dict]) -> dict:
    total = null = hold = 0.
    for before, after in zip(publications, publications[1:]):
        if not isinstance(before['practical'], dict) or not isinstance(before['display'], dict):
            raise ValueError('公開DTOが文字列の旧未検収ログです。構造化記録を修正した候補で再取得してください。')
        elapsed = after['published_at']-before['published_at']
        total += elapsed
        null += elapsed*(before['practical'].get('p1_win_probability') is None)
        hold += elapsed*(before['display'].get('status') == 'hold')
    return dict(seconds=total, null_seconds=null, hold_seconds=hold,
        null_rate=null/total if total else None, hold_rate=hold/total if total else None)


def feature_summary(results: list[dict]) -> dict:
    costs, grouped, bad, valid, cache = [], defaultdict(list), 0, 0, [0, 0]
    for result in results:
        timing = dict(result['timings'])
        for key, value in result['state_timings'].items():
            timing[key] = timing.get(key, 0)+value
        cost = sum(timing.get(k, 0) for k in ('state_ms','ledger_ms','search_ms','feature_only_ms'))
        costs.append(cost)
        row, audit = result.get('row'), result.get('search_audit', {})
        cache[0] += audit.get('cache_hits', 0)
        cache[1] += audit.get('cache_misses', 0)
        grouped['cold' if audit.get('cache_misses') else 'warm'].append(cost)
        grouped['candidates='+str(audit.get('completion_candidates'))].append(cost)
        if row is None:
            continue
        valid += 1
        bad += any(v >= 6 for k,v in row.items() if k.endswith(('_p1','_p2'))
                   and k.replace('_p1','').replace('_p2','') in ('search_trunc_empty','ta_trunc_empty')
                   and isinstance(v, (float,int)))
        for side in ('p1','p2'):
            grouped['requested_H='+str(row.get('ta_req_h_'+side))].append(cost)
    return dict(feature_ms=stats(costs), grouped_feature_ms={k:stats(v) for k,v in grouped.items()},
        truncation_empty_ge6=dict(numerator=bad, denominator=valid, rate=bad/valid if valid else None),
        search_cache=dict(hits=cache[0], misses=cache[1], hit_rate=cache[0]/sum(cache) if sum(cache) else None))


def resource_summary(rows: list[dict], ama_pid: int | None = None) -> dict:
    utilization, ama, previous = [], [], None
    for row in rows:
        if previous:
            dt = row['monotonic']-previous['monotonic']
            old = {r['pid']:r for r in previous.get('processes',[])}
            ama_ids = {ama_pid} | {r['pid'] for r in row.get('processes',[]) if r.get('parent') == ama_pid}
            delta = [(r['pid'],r['name'],r.get('cpu_sec',0)-old.get(r['pid'],{}).get('cpu_sec',r.get('cpu_sec',0)))
                     for r in row.get('processes',[]) if r['name'].lower() != 'obs64.exe']
            utilization.append(sum(max(0,cpu) for _,_,cpu in delta)/dt*100)
            ama.append(sum(max(0,cpu) for pid,name,cpu in delta if pid in ama_ids or name == 'ama_reference.exe')/dt*100)
        previous = row
    return dict(cpu_percent_one_core=stats(utilization), ama_cpu_percent_one_core=stats(ama),
        ama_worker_pid=ama_pid, ama_includes_input_worker=ama_pid is not None,
        minimum_free_gib=min((r['free_gib'] for r in rows if 'free_gib' in r), default=None),
        peak_tree_rss_bytes=max((sum(p.get('rss_bytes',0) for p in r.get('processes',[]) if p['name'].lower() != 'obs64.exe') for r in rows),default=None),
        gpu_samples=sum(r.get('gpu',{}).get('available',False) for r in rows))


def summarize(root: Path) -> dict:
    rows = read(root/'b3_time_gate.jsonl')
    results = [r for r in rows if r['kind']=='result']
    publications = publication_rows(rows)
    ready = [r for r in rows if r['kind']=='evaluation_ready']
    by_frame = {(r['game'],r['frame']):r for r in results if r['accepted']}
    latencies, seen = [], set()
    for pub in publications:
        key = (pub['game'],pub['state_frame'])
        source = by_frame.get(key)
        if key not in seen and source and source.get('captured_at') is not None:
            latencies.append((pub['published_at']-source['captured_at'])*1000)
            seen.add(key)
    rates = time_rates(publications)
    waits = [r['state_wait_ms']+r['compute_wait_ms'] for r in results]
    state_waits = [r['state_wait_ms'] for r in read(root/'b3_state_timing.jsonl')]
    out = dict(status='REVIEW_REQUIRED', duration_ok=rates['seconds']>=1800,
        publication_latency_ms=stats(latencies), wait_ms=stats(waits),
        waits_over_1s=sum(w>1000 for w in waits), accepted=sum(r['accepted'] for r in results),
        all_state_wait_ms=stats(state_waits), all_state_waits_over_1s=sum(w>1000 for w in state_waits),
        discarded=sum(not r['accepted'] for r in results), time_rates=rates,
        parent_queue_start=ready[0]['parent_queue'] if ready else None,
        parent_queue_end=ready[-1]['parent_queue'] if ready else None,
        parent_queue_max=max((r['parent_queue'] for r in ready),default=None),
        ama_request_to_evaluation_ms=stats([r['request_to_display_ms'] for r in read(root/'ama_display.jsonl') if 'request_to_display_ms' in r]),
        **feature_summary(results), resources=resource_summary(read(root/'b3_resources.jsonl'),
            next((r['pid'] for r in read(root/'ama_display.jsonl') if r.get('kind')=='worker'),None)))
    out['review_notes'] = ['公開後のJSONL monotonicで測定。原票published_atは公開処理入口。ブラウザpaintは含まない。',
        '時間門の正式採否はPC1 Claude。30分未満・cold/warm/低速再生未完を合格扱いしない。',
        '7連鎖以上の真値層は原票と動画を照合して別集計。予測連鎖数で代用しない。']
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(summarize(args.output),stream,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    main()
