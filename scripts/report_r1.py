"""R1の事前登録条件だけで採否を決め、母数と対象場面を保存する。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts import report_e34b as previous
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/r1')
Q_MAX = .509977
ZENCHI_MIN = .9167
FALSE_MAX = 1/28
SCENE_START = 2579.066
SCENE_OFFSET = 180.


def scene(mode: str) -> dict:
    """レビュー開始3分の実表示と、同場面の2P≤5%初到達を分ける。"""
    path = previous.directory(mode, 'review')/'display.npz'
    times, values = displayed(path)
    index = int(np.argmin(np.abs(times-(SCENE_START+SCENE_OFFSET))))
    raw = np.load(path)
    return dict(t_sec=float(times[index]), p2_display=float(1-values[index]),
                p2_selected=float(1-raw['display_p1'][index]), first_5percent_sec=first(times, values))


def main() -> None:
    """認識と本番評価の全条件を満たした場合だけレビュー生成を許す。"""
    previous.OUT = OUT
    recognition = json.loads((OUT/'RECOGNITION_METRICS.json').read_text())
    values = {mode: previous.metrics(mode) for mode in ('off', 'on')}
    # E34b独自の合否を混在させず、R1の事前登録条件だけを公開する。
    for value in values.values():
        value.pop('gates', None)
        value.pop('candidate', None)
    after = values['on']
    same_cohort = all(values['off'][key]['frames'] == after[key]['frames'] for key in ('q', 'zenchi'))
    gates = dict(cohort=same_cohort, q=after['q']['log_loss'] <= Q_MAX,
                 zenchi=after['zenchi']['agreement'] >= ZENCHI_MIN,
                 deaths=after['deaths']['unlabelled'] == 0 and
                 after['deaths']['false']/max(1, after['deaths']['total']) <= FALSE_MAX)
    all_gates = dict(recognition['gates'], e=all(gates.values()))
    result = dict(recognition=recognition, metrics=values, evaluation_gates=gates,
                  gates=all_gates, passed=all(all_gates.values()),
                  scenes={mode: scene(mode) for mode in values},
                  adoptions={mode: previous.adoptions(mode) for mode in values})
    save_json(OUT/'SUMMARY.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
