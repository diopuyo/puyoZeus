"""R1bの事前登録b/c/d/eだけで採否を決め、OFF/R1/R1bを併記する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts import report_r1 as prior
from scripts.r1_measure_helpers import lines
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/r1b')
SCENE_TIME = 2759.05


def counter(root: Path, mode: str) -> dict:
    """3:00以前の最後の実評価に2Pの反撃予測が採用されたかを見る。"""
    events = lines(root/mode/'review/events.jsonl')
    values = [v for e in events for v in e['values']
              if v['source'] == 'S3_landing' and v['t_sec'] <= SCENE_TIME]
    latest = max(values, key=lambda v: v['t_sec']) if values else {}
    predictions = [r for r in latest.get('prefire_prediction', []) if r['side'] == '2P']
    return dict(t_sec=latest.get('t_sec'), adopted=bool(predictions), predictions=predictions,
                incoming=latest.get('incoming'), dead_sides=latest.get('dead_sides'))


def main() -> None:
    """旧結果を書き換えず、新ONと同じ母数・固定閾値で判定する。"""
    before = json.loads(Path('logs/r1/SUMMARY.json').read_text())
    recognition = json.loads((OUT/'RECOGNITION_METRICS.json').read_text())
    prior.previous.OUT = OUT
    after = prior.previous.metrics('on')
    after.pop('gates', None)
    after.pop('candidate', None)
    same = all(before['metrics']['off'][k]['frames'] == after[k]['frames'] for k in ('q', 'zenchi'))
    evaluation = dict(cohort=same, q=after['q']['log_loss'] <= prior.Q_MAX,
        zenchi=after['zenchi']['agreement'] >= prior.ZENCHI_MIN,
        deaths=after['deaths']['unlabelled'] == 0 and
        after['deaths']['false']/max(1, after['deaths']['total']) <= prior.FALSE_MAX)
    gates = dict(recognition['gates'], e=all(evaluation.values()))
    result = dict(recognition=recognition, evaluation_gates=evaluation, gates=gates,
        passed=all(gates.values()), metrics=dict(off=before['metrics']['off'],
        r1=before['metrics']['on'], r1b=after), scenes=dict(off=before['scenes']['off'],
        r1=before['scenes']['on'], r1b=prior.scene('on')),
        adoptions=dict(off=before['adoptions']['off'], r1=before['adoptions']['on'],
                       r1b=prior.previous.adoptions('on')),
        counter=dict(off=counter(Path('logs/r1'), 'off'), r1=counter(Path('logs/r1'), 'on'),
                     r1b=counter(OUT, 'on')))
    save_json(OUT/'SUMMARY.json', result)
    print(json.dumps(dict(gates=gates, passed=result['passed'], metrics=result['metrics'],
                         scenes=result['scenes'], counter=result['counter']), ensure_ascii=False))


if __name__ == '__main__':
    main()
