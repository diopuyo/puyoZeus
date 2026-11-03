"""原映像で確認した入力差だけを変える反実仮想。実行時ロジックは変更しない。"""
from __future__ import annotations

import json

import numpy as np

from scripts.inspect_e24_saved_20260928 import OUT, frames, TIMES
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.board import Board
from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_event_multilanding import prove_multilanding

Q_TIME = 832.7333333333333
F_TIME = 490.8666666666667
# 原映像と突合した0起点座標。認識器へ書き戻すための補正ではない。
Q_CELL_CHECKS = ((7, 2, 5, 3), (7, 3, 5, 4))


def probe(source: str, stamp: float, idx: int) -> dict:
    """時点を固定し、元入力と局所変更入力の証明を同条件で比較する。"""
    rows = json.loads((OUT/f'{source}_runtime.json').read_text())
    row = next(r for r in rows if abs(r['t_sec']-stamp) < 1e-8)
    side = row['sides'][idx]
    board = Board.from_list(side['response_board'])
    projection = ExchangeLandingProjection(multi_landing_death=True)

    def run(current: Board, amount: int) -> dict:
        return prove_multilanding(current, tuple(side['queue']), amount, side['hands'],
            row['elapsed'], projection._optimistic_response, side['credit'])

    original = run(board, side['incoming'])
    assert original == row['multi_landing'][idx]
    result = dict(t_sec=stamp, original=original, elapsed=row['elapsed'],
                  queue=side['queue'], hands=side['hands'], incoming=side['incoming'])
    result['optimistic_before'] = projection._optimistic_response(
        board, np.array(side['queue']), side['hands'], row['elapsed'])
    if source == 'q_7gc4TgFig':
        for r, c, old, new in Q_CELL_CHECKS:
            assert board._grid[r, c] == old
            board._grid[r, c] = new
        result.update(cell_checks=Q_CELL_CHECKS, corrected_two_cells=run(board, side['incoming']))
        result['optimistic_after'] = projection._optimistic_response(
            board, np.array(side['queue']), side['hands'], row['elapsed'])
    else:
        # 前交換の30,500点全量と、その前の1,440点を控除した量の両端を感度分析する。
        result['pending_sensitivity'] = {n: run(board, n) for n in (162, 182)}
    return result


def main() -> None:
    """証拠画像を追加し、再生終了後に入力感度を測る。"""
    TIMES.update(q_7gc4TgFig=(832.4, 833., 833.7, 834., 834.6, 835.3, 835.533333, 836., 837.),
                 fcXG83vInDY=(477., 477.833333, 478.4, 480., 482., 488., 491.6, 492.366667, 499.8))
    for source in TIMES:
        frames(source, '_detail')
    for source, stamp, idx in (('q_7gc4TgFig', Q_TIME, 1), ('fcXG83vInDY', F_TIME, 0)):
        if (OUT/f'{source}_runtime.json').exists():
            save_json(OUT/f'{source}_input_probe.json', probe(source, stamp, idx))


if __name__ == '__main__':
    main()
