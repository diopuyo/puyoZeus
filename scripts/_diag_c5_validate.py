"""原票一致と受け量だけを変更した診断入力を検査する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._diag_c5_inspect import BASE, OUT, guard, save
from src.board import Board
from src.post_counter_death_bound import prove_post_counter


def read(path: Path) -> object:
    """UTF-8原票を読む。"""
    return json.loads(path.read_text())


def normalized(value: object) -> object:
    """実行時間だけを比較対象から除く。"""
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k != 'elapsed_sec'}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def main() -> None:
    """元D-v2cの全監査行と初回証明を比較する。"""
    guard()
    old = read(BASE / 'v2c/short/c5/post_counter_bound_audit.json')['rows']
    new = read(OUT / 'replay/post_counter_bound_audit.json')['rows']
    prefix = new[:len(old)]
    assert normalized(old) == normalized(prefix)
    original_events = [json.loads(line) for line in (BASE / 'v2c/short/c5/events.jsonl').open()]
    replay_events = [json.loads(line) for line in (OUT / 'replay/events.jsonl').open()]
    old_values = [v for event in original_events for v in event['values']]
    new_values = [v for event in replay_events for v in event['values']]
    assert old_values == new_values[:len(old_values)]
    evidence = read(OUT / 'first_proof_inputs.json')
    board = Board.from_list(evidence['context']['replies'][1])
    actual = next(r for r in read(OUT / 'stable_boards.json') if r['side'] == 2 and r['t'] >= 1731)
    assert actual['board'] == evidence['context']['replies'][1]
    audit = next(row for row in old if row['dead'])
    elapsed = evidence['safety']['elapsed']['(1, 4)']
    results = {str(amount): prove_post_counter(board, tuple(audit['queue']), amount,
               audit['hands'], elapsed, tuple(audit['palette'])) for amount in (77, 0)}
    assert results['77']['dead'] and not results['0']['dead']
    validation = dict(audit_prefix_rows=len(old), audit_prefix_equal=True,
        event_value_prefix_rows=len(old_values), event_value_prefix_exact_equal=True,
        excluded='証明実行時間elapsed_secのみ', post_chain_board_exact_equal=True,
        received_only_counterfactual=results, available_bytes=guard(), nice=10)
    save('validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()
