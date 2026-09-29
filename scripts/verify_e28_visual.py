"""2755.017秒の目視転記を独立検算する。認識器・評価器には反映しない。"""
from pathlib import Path
import numpy as np
from src.board import Board
from src.chain import ChainSimulator
from src.exchange_midchain_completion import remaining
from src.exchange_hidden_row_death import enumerate_hidden
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from scripts.run_e3_exchange_eval_20260926 import save_json

VISIBLE_SEC = 2755.016666666667
# 1Pの9段目黄色、2Pの3段目赤は点滅消去中。以下はその色群を除いた転記。
P1 = [[0]*6 for _ in range(6)] + [
    [0,0,0,3,0,4], [0,0,0,5,5,4], [0,0,0,3,3,4],
    [0,0,0,4,4,3], [0,0,1,3,4,5], [3,0,0,1,1,1], [0,0,1,5,5,5]]
P2 = [[10,0,0,0,0,10], [5,0,0,0,0,3], [1,0,0,0,0,1],
    [3,0,0,0,0,1], [4,3,0,0,0,1], [5,4,0,0,3,4],
    [5,5,3,0,4,4], [3,3,3,4,9,5], [5,4,4,3,4,3],
    [3,3,3,4,3,4], [1,4,1,3,5,4], [1,1,4,1,1,1], [4,4,1,5,5,5]]


def main() -> None:
    """画面から消去後を手で読んだ上限であり、本経路の採用成功とは数えない。"""
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    p1, p2 = Board(), Board()
    p1._grid, p2._grid = np.asarray(P1, dtype=p1._grid.dtype), np.asarray(P2, dtype=p2._grid.dtype)
    # 目視で消去中の色群を除いたため生じた空洞だけ、診断上の重力で落とす。
    for col in range(p1._grid.shape[1]):
        values = p1._grid[:, col][p1._grid[:, col] != 0]
        p1._grid[:, col] = 0
        if len(values):
            p1._grid[-len(values):, col] = values
    first = remaining(p1, 9, 35920, simulator)
    replies, trials = enumerate_hidden(p2, (1,3,4,5), 3, 1320, simulator)
    assert first and first['score'] == 87720 and first['count'] == 13
    assert len(replies) == trials == 25
    assert {r['score'] for r in replies} == {75440}
    assert {r['count'] for r in replies} == {13}
    save_json(Path('logs/e28/VISUAL_VERIFICATION.json'), dict(t_sec=VISIBLE_SEC,
        source_frame='logs/e28/frames/2755.017.jpg', p1_pop=9, p2_pop=3,
        method='画面の点滅消去群を除いた手転記と重力。評価への反映なし。',
        p1_after_pop=P1, p2_after_pop=P2, p1_prediction=first,
        p2_trials=trials, p2_score_max=75440, p2_chain_count=13,
        score_gap=87720-75440, send_p1=87720//70, send_p2=75440//70,
        net_before_previous_cancel=87720//70-75440//70,
        previous_cancel=5, net_after_previous_cancel=171))
    print('目視転記の検算: 87720対75440、13連鎖同士、純受け171個', flush=True)


if __name__ == '__main__':
    main()
