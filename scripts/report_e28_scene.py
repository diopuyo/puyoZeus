"""診断CSVに日本語の列説明と、根拠を持つ場面要約を付ける。"""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.measure_e23_display_20260928 import displayed, first

OUT = Path('logs/e28')


def terminal_times() -> dict:
    """低確率表示と、実映像の死確定・予測による回避不能死を混同しない。"""
    data = np.load('logs/e27/on/review/display.npz')
    times = {}
    for source in ('confirmed_death', 'unavoidable_death'):
        selected = data['t_sec'][(data['t_sec'] >= 2755) & (data['source'] == source)]
        times[source] = float(selected[0]) if len(selected) else None
    return times


def main() -> None:
    """一枚診断と実採用を区別し、システムの死確定と確率低下も分ける。"""
    raw = json.loads((OUT/'SCENE_RAW.json').read_text())
    samples = json.loads((OUT/'SINGLE_SAMPLE_DIAGNOSTIC.json').read_text())
    scene = [r for r in raw if 2755 <= r['t_sec'] <= 2771]
    checks = []
    for sample in samples:
        count = sample['formula'][0]+1
        later = next((r for r in raw if r['t_sec'] > sample['t_sec']
                      and r['formula'] and r['formula'][0] == count), None)
        matching = [v for v in sample['options'] if later and v['prefix'].get(str(count)) == later['formula'][1]]
        checks.append(dict(t_sec=sample['t_sec'], trials=sample['trials'],
            candidates=len(sample['options']), next_sec=later['t_sec'] if later else None,
            next_count=count, matching=len(matching),
            maximum_score=max((v['score'] for v in matching), default=None), adopted=False))
    evaluated = [r for r in scene if r['last'].get('t_sec') == r['t_sec']]
    shown = first(*displayed(Path('logs/e27/on/review/display.npz')))
    raw_low = next(r['t_sec'] for r in raw if r['p1'] is not None and r['p1'] >= .95)
    result = dict(scene_observations=len(scene), scene_landing_evaluations=len(evaluated),
        multi_reasons=dict(Counter(r['last']['multi_landing'][1]['reason'] for r in evaluated)),
        midchain_adopted=sum(r['midchain_status'] == '採用中' for r in scene),
        hidden_adopted=sum(r['hidden_status'] == '採用中' for r in scene),
        multi_searches=sum('rounds' in r['last']['multi_landing'][1] for r in evaluated),
        unavoidable_death_sec=next((r['t_sec'] for r in raw if r['source'] == 'unavoidable_death'), None),
        first_completion_certain=next(r['t_sec'] for r in scene if r['t_sec'] > 2755
            and r['last'].get('completion_certain', [False, False])[1]),
        exchange_closed_sec=next(r['t_sec'] for r in raw if r['t_sec'] > 2755 and not r['exchange_open']),
        first_raw_p2_le_5=raw_low, first_display_p2_le_5=shown,
        first_visual_sec=2755.016666666667, display_delay_sec=shown-2755.016666666667,
        single_sample_checks=checks, terminal_times=terminal_times())
    save_json(OUT/'SCENE_CONCLUSION.json', result)
    for name in ('SCENE.csv', 'SCENE_CONTEXT.csv'):
        with (OUT/name).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        # 未計算の既存内部値0を、外向けの相殺必要量0とは解釈しない。
        for row in rows:
            if not row['p2_near_future_ojama']:
                row['p2_required_cancel_ojama'] = ''
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
