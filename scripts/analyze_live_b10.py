"""B9の保存結果だけを再集計し、B10の根拠を固定する。動画は再処理しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_live_b9 import report, lines, samples, available
from scripts.measure_live_b6 import save
from scripts.measure_live_b9 import START_SEC, END_SEC


def diagnose(path: Path) -> dict:
    events = lines(path/'faults.jsonl')
    begin, end = [event['at'] for event in events]
    original = json.loads((path/'result.json').read_text())['analysis']
    sent = samples(path)
    evaluations = {r['frame']: r for r in json.loads((path/'evaluations.json').read_text())}
    exposed = []
    for row in sent:
        if not available(row) or not begin <= row['received_at'] < end:
            continue
        payload = row['payload']
        value = evaluations[payload['timing']['source_available_frame']]
        exposed.append(dict(received_at=row['received_at'], after_fault_sec=row['received_at']-begin,
            stream_seq=payload['identity']['stream_seq'], **{key: value[key] for key in
            ('frame', 't_sec', 'game', 'captured_at', 'recognized_at', 'evaluated_at',
             'source', 'probability', 'state1', 'state2', 'queue_depth')}))
    return dict(events=events, original={k: v for k, v in original.items() if k != 'first_after_recovery'},
                exposures=exposed)


def reanalyze(source: Path, output: Path) -> dict:
    long = report(source/'long', START_SEC, END_SEC)
    long['needs_review'] = [key for key, value in long['trends'].items() if value['strictly_increasing']]
    long['passed'] = (not long['needs_review'] and long['games']['count_matches'] and
        all(row['first_probability_delay_sec'] is not None for row in long['games']['rows']))
    long['source'] = str(source/'long')
    save(output/'long_reanalysis.json', long)
    diagnoses = {case: diagnose(source/case) for case in ('resolution', 'repeat')}
    save(output/'diagnosis.json', diagnoses)
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('logs/live_b9'))
    parser.add_argument('--output', type=Path, default=Path('logs/live_b10'))
    options = parser.parse_args()
    result = reanalyze(options.source, options.output)
    print(json.dumps(dict(windows=result['windows'], games=result['games'], needs_review=result['needs_review']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
