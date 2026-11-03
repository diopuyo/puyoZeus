"""元監査との一致と黄色二セル欠落の反事実を検証する。"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._diag_mia7_inspect import BASE, OUT, guard, save
from src.board import Board
from src.chain import ChainSimulator
from src.scoring import calculate_step_score
from src.post_counter_death_bound import prove_post_counter
from src.exchange_pending_ledger import PendingLedger

TARGET, EPSILON, STAGE_OFFSET = 691.4333333333333, 0.0001, 2
YELLOW, COLUMN, ROWS = 4, 4, (7, 8)
RATE = 70


def read(path: Path) -> Any:
    """圧縮原票もそのまま読み取る。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def normalized(value: Any) -> Any:
    """処理時間だけを一致判定から除く。"""
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k != 'elapsed_sec'}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def simulation(board: Board, offset: int = 0) -> dict:
    """画像から独立に転記した二セルと段番号だけを診断用に変える。"""
    result = ChainSimulator(exclude_hidden_row_from_pop=True).simulate(board)
    steps = []
    for step in result.steps:
        step.chain_index += offset
        steps.append(asdict(calculate_step_score(step)))
    return dict(count=result.chain_count, score=sum(s['score'] for s in steps),
                steps=steps, board=result.final_board._grid.tolist())


def ledger_variant(evidence: dict, score: int) -> list[int]:
    """履歴・既着弾を固定し、対象連鎖の点数だけを改訂する。"""
    value = evidence['safety']['ledger']
    ledger = PendingLedger()
    ledger.game = value['game']
    ledger.events = [(kind, tuple(key), amount) for kind, key, amount in value['events']]
    ledger.totals = {tuple(map(int, k.strip('()').split(','))): tuple(v) for k,v in value['totals'].items()}
    ledger.dropped = tuple(value['dropped'])
    totals = dict(ledger.totals)
    totals[(1, 12)] = (score // RATE, True)
    ledger.observe(ledger.game, totals, ledger.dropped)
    return ledger.pending


def main() -> None:
    """再現と各入力の独立差し替え結果を保存する。"""
    guard()
    original = BASE / 'v2e/evidence/full/mia8KCjr52g'
    old = read(original / 'post_counter_bound_audit.json.gz')['rows']
    new = read(OUT / 'replay/post_counter_bound_audit.json')['rows']
    old = [r for r in old if r['t_sec'] <= 706]
    assert normalized(old) == normalized(new)
    evidence = read(OUT / 'first_proof_inputs.json')
    audit = next(r for r in new if abs(r['t_sec']-TARGET) < EPSILON and r['side'] == '2P')
    board = Board.from_list(evidence['latest'][1]['board'])
    corrected = board.copy()
    for row in ROWS:
        assert corrected._grid[row, COLUMN] == 0
        corrected._grid[row, COLUMN] = YELLOW
    sims = dict(original=simulation(board), two_cells=simulation(corrected),
                two_cells_and_stage=simulation(corrected, STAGE_OFFSET))
    actual = Board.from_list(sims['two_cells']['board'])
    predicted = Board.from_list(evidence['context']['replies'][1])
    results = {}
    for name, grid, pending in (('original', predicted, 118), ('receive_zero', predicted, 0),
                                ('actual_final_only', actual, 118), ('actual_final_zero', actual, 0)):
        results[name] = prove_post_counter(grid, tuple(audit['queue']), pending,
            audit['hands'], evidence['safety']['elapsed']['(1, 12)'], tuple(audit['palette']))
    assert sims['original']['score'] == 40 and sims['two_cells_and_stage']['score'] == 20260
    assert results['original']['dead'] and not results['receive_zero']['dead']
    ledger_results = {str(score): ledger_variant(evidence, score) for score in (40, 8860, 20260)}
    actual_row = next(r for r in read(OUT / 'stable_boards.json') if r['side'] == 2 and abs(r['t']-699.2666666666667) < EPSILON)
    assert actual_row['board'] == sims['two_cells']['board']
    save('validation.json', dict(audit_rows=len(new), audit_equal=True,
        excluded='elapsed_secのみ', first_audit=audit, simulations=sims, counterfactuals=results,
        projected_ledger=ledger_results, corrected_board_matches_actual=True,
        nice=10, available_bytes=guard()))
    print(json.dumps(dict(audit=audit, simulations=sims, counterfactuals=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
