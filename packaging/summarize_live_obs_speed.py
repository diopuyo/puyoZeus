"""配布設定の更新処理量・公開遅延・独立ama遅延を別々に集計する。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_live_obs_time_gate import read, stats, resource_summary, publication_rows

RATIO = 1.6473854230


def interval_ms(values: list[tuple]) -> float:
    end, cost = float('-inf'), 0.
    for start, finish in sorted(values):
        cost += max(0.,finish-max(start,end))
        end = max(end,finish)
    return cost*1000


def b3_cost(root: Path) -> dict:
    cost = defaultdict(float)
    for row in read(root/'b3_state_timing.jsonl'):
        cost[row['frame']] += row.get('state_ms',0)+row.get('ledger_ms',0)
    for row in read(root/'b3_updates.jsonl'):
        cost[row['frame']] += sum(row.get(k,0) for k in ('search_ms','feature_only_ms','inference_ms'))
    return cost


def combined(root: Path) -> list[dict]:
    recognition = read(root/'spool/frame_timing_recognition.jsonl')
    b3, v6, v49f, intervals = b3_cost(root), b3_cost(root/'v49f_b3_v6'), defaultdict(float), defaultdict(list)
    values = read(root/'v49f_updates.jsonl')
    updated = {r['id'] for r in values if r['value'] is not None}
    for row in values:
        v49f[row['id']] += row['compute_ms']
    for role in ('evaluation','publication'):
        for row in read(root/f'spool/frame_timing_{role}.jsonl'):
            intervals[row['frame']].append((row['started_at'],row['finished_at']))
    result = []
    for row in recognition:
        frame = row['frame']
        if frame is None:
            continue
        parts = dict(recognition=row['wall_ms'],b3_m3=b3[frame],v49f=v49f[frame],
            v49f_baseline_v6=v6[frame],display_dispatch=interval_ms(intervals[frame]))
        result.append(dict(frame=frame,t_sec=row['t_sec'],v49f_updated=frame in updated,
            measured_ms=parts,low_spec_ms={k:v*RATIO for k,v in parts.items()},
            total_low_spec_ms=sum(parts.values())*RATIO,recognition_stages=row['stages']))
    return result


def ama_delay(root: Path, polls: list[dict]) -> dict:
    jobs = {(r['game'],r['result']['input_sec']):r for r in read(root/'ama_updates.jsonl') if 'value' in r.get('result',{})}
    seen, delay = set(), []
    for poll in polls:
        value = poll['snapshot'].get('evaluations',{}).get('ama_plain',{})
        if value.get('p1') is None:
            continue
        # 公開試合番号は内部番号と異なるため、時刻一致と依頼以前でないことを確認する。
        candidates = [(key,job) for key,job in jobs.items() if key[1]==value['input_sec'] and job['requested_at']<=poll['at']]
        if not candidates:
            continue
        key, job = max(candidates,key=lambda x:x[1]['requested_at'])
        if key not in seen:
            delay.append((poll['at']-job['requested_at'])*1000)
            seen.add(key)
    return dict(request_to_http_poll_ms=stats(delay),poll_interval_ms=100,
        request_to_evaluation_ms=stats([r['request_to_display_ms'] for r in read(root/'ama_display.jsonl') if 'request_to_display_ms' in r]),
        native_and_conversion_cpu_sec=sum(r['result'].get('cpu_sec',0) for r in jobs.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    receipt = json.loads((args.run/'receipt.json').read_text())
    assert receipt['benchmark'], 'smokeの時間を速度測定結果として使わない'
    args.out.mkdir(exist_ok=False)
    root, groups = args.run/'output', {}
    rows = combined(root)
    for name, selected in (('all_frames',rows),('v49f_update_frames',[r for r in rows if r['v49f_updated']])):
        groups[name] = {k:stats([r['low_spec_ms'][k] for r in selected]) for k in rows[0]['low_spec_ms']}
        groups[name]['total'] = stats([r['total_low_spec_ms'] for r in selected])
    publications = publication_rows(read(root/'b3_time_gate.jsonl'))
    summary = dict(ratio=RATIO,**groups,ama=ama_delay(root,read(args.run/'polls.jsonl')),
        resources=resource_summary(read(root/'b3_resources.jsonl'),
            next((r['pid'] for r in read(root/'ama_display.jsonl') if r.get('kind')=='worker'),None)),
        capture_to_publication_ms=stats([(r['published_at']-r['captured_at'])*1000 for r in publications]),
        boundary='並列処理量の和と実際の公開遅延を分離。ブラウザpaint未計測。')
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')


if __name__ == '__main__':
    main()
