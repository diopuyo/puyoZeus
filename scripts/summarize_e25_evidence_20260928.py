"""最終再生の場面別原票をCSV化し、E24の誤発火2件と比較する。"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e25')
CASES = (('q_7gc4TgFig', 832.7333333333333, 1),
         ('fcXG83vInDY', 490.8666666666667, 0))


def csv_rows(path: Path, rows: list[dict]) -> None:
    """JSONと同じ評価行を保存し、未実行の応手量を0に置き換えない。"""
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scene() -> dict:
    """実行フラグで評価契機が変わるため、E23の90行を強制的に流用しない。"""
    values = json.loads((OUT/'on/review/scene_evaluations.json').read_text())
    rows = [dict(t_sec=v['t_sec'], incoming=v['incoming'][1],
        certain=v['completion_certain'][1], dead='2P' in v['dead_sides'],
        reason=v['multi_landing'][1]['reason'], state_safety=v['state_safety'][1],
        proof=json.dumps(v['multi_landing'][1], ensure_ascii=False)) for v in values]
    csv_rows(OUT/'ZENCHI_SCENE_TRACE.csv', rows)
    audit = json.loads((OUT/'on/review/completion_audit.json').read_text())
    completion = [r for r in audit if r['game'] == 3 and r['side'] == '2P' and r['chain_id'] == 9]
    csv_rows(OUT/'ZENCHI_COMPLETION_TRACE.csv', completion)
    return dict(evaluations=len(rows), reasons=dict(Counter(r['reason'] for r in rows)),
        recovered=sum(r['recovered'] for r in completion))


def case_checks() -> list[dict]:
    """同じ動画・受け側・時刻で、純受けと死亡判定をE23から比較する。"""
    rows = []
    for source, stamp, idx in CASES:
        for version in ('e23', 'e25'):
            path = Path(f'logs/{version}/on/renders/{source}/on/events.jsonl')
            events = [json.loads(line) for line in path.read_text().splitlines()]
            values = [v for e in events for v in e['values'] if 'multi_landing' in v
                      and abs(v['t_sec']-stamp) < 1e-7]
            assert len(values) == 1, (source, version, values)
            value = values[0]
            rows.append(dict(version=version, source=source, t_sec=stamp, side=f'{idx+1}P',
                incoming=value['incoming'][idx], dead=f'{idx+1}P' in value['dead_sides'],
                reason=value['multi_landing'][idx]['reason'],
                safety=value.get('state_safety', [None, None])[idx]))
    csv_rows(OUT/'PREVIOUS_ERROR_CHECKS.csv', rows)
    return rows


def main() -> None:
    """時刻丸め前の結果と検証済み側ラベルを残す。"""
    result = dict(scene=scene(), previous_errors=case_checks())
    save_json(OUT/'EVIDENCE_SUMMARY.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
