"""保存した証明入力への局所的な反事実を計算する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts._diag_set2_falsedeath_extract import OUT
from src.board import Board
from src.post_counter_death_bound import prove_post_counter

# 実画面で判別できるセルだけ。隠し段やエフェクトで読めないセルは断定しない。
VISIBLE_EDITS = {13: ((1, 1, 0),), 43: ((3, 2, 2),),
                 45: ((1, 2, 0),), 58: ((1, 2, 0),)}
STANDARD_RATE_ELAPSED = 0.0


def compact(proof: dict) -> dict:
    """局所証明の結論と探索量を残す。"""
    return {k: proof[k] for k in ('dead', 'reason', 'nodes', 'pruned')}


def case_proofs(case: dict) -> dict:
    """元入力、可視誤セル訂正、未知ツモ許容を独立に比較する。"""
    part, stamp = case['label']['part'], case['first_sec']
    trace = json.loads((OUT / 'baseline' / part / f'trace_{stamp:.6f}.json').read_text())
    idx = int(case['side'] == '2P')
    board = Board.from_list(trace['context']['replies'][idx])
    queue = tuple(trace['latest'][idx]['queue'])
    incoming = trace['context']['incoming'][idx]
    hands = case['first_value']['hands'][idx]
    palette = tuple(sorted({int(c) for row in board._grid.tolist() for c in row if 1 <= c <= 5}))
    variants = [('original', board, queue, hands), ('unknown_queue', board, (0, 0, 0, 0), hands),
                ('plus_two_hands', board, queue, hands + 2)]
    if case['game'] in VISIBLE_EDITS:
        corrected = board.copy()
        for row, col, color in VISIBLE_EDITS[case['game']]:
            corrected.set(row, col, color)
        variants.append(('visible_cell_corrected', corrected, queue, hands))
    proofs = {}
    for name, current, supply, budget in variants:
        proof = prove_post_counter(current, supply, incoming, budget, STANDARD_RATE_ELAPSED, palette)
        proofs[name] = dict(board_dead=current.is_dead(), **compact(proof))
    return dict(game=case['game'], incoming=incoming, hands=hands, queue=queue,
                palette=palette, visible_edits=VISIBLE_EDITS.get(case['game'], ()), proofs=proofs)


def main() -> None:
    """全初回は開始96秒未満なので同じ70点換算で比較する。"""
    path = OUT / 'LOCAL_PROOFS.json'
    if path.exists():
        return
    cases = json.loads((OUT / 'cases.json').read_text())
    rows = [case_proofs(c) for c in cases]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
