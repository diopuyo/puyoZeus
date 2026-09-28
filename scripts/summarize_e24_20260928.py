"""E24の原票・再生・感度分析を突合し、結論の数値と根拠を固定する。"""
from __future__ import annotations

from collections import Counter
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.inspect_e24_saved_20260928 import OUT, SIDES, WINDOWS, event_path, write_csv
from scripts.run_e3_exchange_eval_20260926 import save_json

EPSILON = 1e-8
MAX_FUNCTION_LINES = 50
FIRST_DEATH = dict(review=2770.616666666667, q_7gc4TgFig=832.7333333333333,
                   fcXG83vInDY=490.8666666666667)


def read(name: str) -> Any:
    """保存済みJSONだけを読む。"""
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def validate_scripts() -> dict:
    """追加スクリプトの構文と1関数50行規約を、実行結果の検査と併せて固定する。"""
    counts = {}
    for path in Path('scripts').glob('*e24*20260928.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        lengths = [node.end_lineno-node.lineno+1 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert max(lengths) <= MAX_FUNCTION_LINES, (path, max(lengths))
        counts[str(path)] = dict(syntax='passed', max_function_lines=max(lengths))
    return counts


def summarize(source: str) -> dict:
    """CSV全行を実再生と照合し、未実行・打切り・死亡を混同せず集計する。"""
    rows = read(f'{source}_runtime.json')
    idx = SIDES[source]
    with (OUT/f'{source}_evaluations.csv').open(encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    assert len(rows) == len(saved)
    for row, flat in zip(rows, saved):
        assert abs(row['t_sec']-float(flat['t_sec'])) < EPSILON
        assert row['multi_landing'][idx]['reason'] == flat['reason']
        assert row['sides'][idx]['incoming'] == int(flat['incoming'])
    first = next(r for r in rows if r['multi_landing'][idx]['dead'])
    assert abs(first['t_sec']-FIRST_DEATH[source]) < EPSILON
    proof = first['multi_landing'][idx]
    return dict(evaluations=len(rows), reasons=dict(Counter(r['reason'] for r in saved)),
        first_dead_sec=first['t_sec'], incoming=first['sides'][idx]['incoming'],
        hands=first['sides'][idx]['hands'], nodes=proof['nodes'],
        death_round=len(proof['rounds']), rounds=[dict(round=n, branches=len(states),
            cancel_min=min(s['cancelled'] for s in states), cancel_max=max(s['cancelled'] for s in states),
            drop_min=min(s['dropped'] for s in states), drop_max=max(s['dropped'] for s in states))
            for n, states in enumerate(proof['rounds'], 1)],
        equivalence=read(f'{source}_equivalence.json')['equivalence'])


def scene() -> None:
    """certain=Falseの直接条件を全90評価に追加し、着弾枝CSVと時刻で結合できる形にする。"""
    rows = []
    for value in read('review_runtime.json'):
        side, proof = value['sides'][1], value['multi_landing'][1]
        chain = [c for c in value['chains'] if c['side'] == '2P'][-1]
        row = dict(t_sec=value['t_sec'], incoming=side['incoming'], hands=side['hands'],
            certain=side['certain'], dead=proof['dead'], stop_reason=proof['reason'],
            death_round=len(proof.get('rounds', [])) if proof['dead'] else None,
            search_executed='nodes' in proof, nodes=proof.get('nodes'),
            busy=value['busy'][1], chain_id=chain['chain_id'],
            predicted_board_present=chain['predicted_final_board'] is not None,
            predicted_score=chain['predicted_final_score'], predicted_count=chain['predicted_chain_count'],
            end_signal_sec=chain['end_signal_sec'], end_confirmed=chain['end_confirmed'])
        for turn, states in enumerate(proof.get('rounds', []), 1):
            row[f'round{turn}_cancel_min'] = min(s['cancelled'] for s in states)
            row[f'round{turn}_cancel_max'] = max(s['cancelled'] for s in states)
        rows.append(row)
    write_csv(OUT/'ZENCHI_SCENE_TRACE.csv', rows)


def main() -> None:
    """全件等価性・原票ハッシュ・入力感度を保存し、実装変更を伴わない診断を再現可能にする。"""
    result = {source: summarize(source) for source in WINDOWS}
    result['diagnostic_validation'] = validate_scripts()
    scene()
    for source in ('q_7gc4TgFig', 'fcXG83vInDY'):
        probe = read(f'{source}_input_probe.json')
        if source == 'q_7gc4TgFig':
            assert not probe['corrected_two_cells']['dead']
            result[source]['input_probe'] = dict(optimistic_before=probe['optimistic_before'],
                optimistic_after=probe['optimistic_after'],
                reason=probe['corrected_two_cells']['reason'], nodes=probe['corrected_two_cells']['nodes'])
        else:
            assert all(not p['dead'] for p in probe['pending_sensitivity'].values())
            result[source]['input_probe'] = {amount: p['reason']
                for amount, p in probe['pending_sensitivity'].items()}
    result['scene_display_e23'] = json.loads(
        (OUT.parent/'e23/SCENE_DISPLAY.json').read_text(encoding='utf-8'))
    result['source_sha256'] = {str(event_path(s)): hashlib.sha256(event_path(s).read_bytes()).hexdigest()
                               for s in WINDOWS}
    save_json(OUT/'SUMMARY.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
