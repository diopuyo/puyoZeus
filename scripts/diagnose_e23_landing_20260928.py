"""E22の分岐を変更せず、指定区間の各着弾評価と入力を保存する。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_virtual_board import land_pending_ojama_onto_board

OUT = Path('logs/e23')
SCENE = (2755., 2771.)
OPTIONS = dict(death_guard=True, confirmed_death_hold=True, death_formula_guard=True)


def capture(projection: Any, overlay: Any, latest: tuple, incoming: list,
            hands: tuple, t_sec: float, value: dict) -> dict:
    """短絡で未評価の条件はnullのまま、独立に観測できる条件も記録する。"""
    replies, credit = projection._receivers(overlay.tracker, latest, incoming)
    boards, replies, certain = projection._death_boards(
        overlay.tracker, tuple(s.board for s in latest), replies, credit)
    sides = []
    for i, board in enumerate(boards):
        landed = land_pending_ojama_onto_board(board, boards[1-i], incoming[i])[0]
        known = projection._known_budget(overlay.tracker, 1-i, t_sec)
        evidence = projection._verified_attack(overlay.tracker, 1-i)
        available, required = value['near_future_send'][i], value['required_cancel'][i]
        margin, optimistic = value['overflow_rows'][i], value['optimistic_send'][i]
        reasons = [('incoming<=0', incoming[i] <= 0), ('landed.is_dead=False', not landed.is_dead()),
            ('certain=False', not certain[i]), ('known_budget=False', not known),
            ('evidence=False', not evidence), ('available>=required', available is not None and available >= required),
            ('margins<1', margin is not None and margin < 1),
            ('optimistic>=required', optimistic is not None and optimistic >= required)]
        sides.append(dict(side=f'{i+1}P', incoming=incoming[i], landed_dead=landed.is_dead(),
            certain=certain[i], known_budget=known, evidence=evidence, required=required,
            available=available, margin=margin, optimistic=optimistic,
            first_blocker=next((k for k, failed in reasons if failed), None), hands=hands[i],
            board=board._grid.tolist(), response_board=replies[i]._grid.tolist(),
            queue=latest[i].queue.tolist(), credit=credit[i]))
    return dict(t_sec=t_sec, game_idx=overlay._game, elapsed=overlay.tracker._score_elapsed,
                source=value['source'], p1=value['p1'], sides=sides)


def main() -> None:
    """E22出力の完全一致も検証し、診断の副作用がないことを確認する。"""
    original, rows = ExchangeLandingProjection._evaluate, []

    def evaluate(self: Any, overlay: Any, snapshot: Any, latest: tuple, incoming: list,
                 hands: tuple, base: dict, t_sec: float) -> dict:
        value = original(self, overlay, snapshot, latest, incoming, hands, base, t_sec)
        if SCENE[0] <= t_sec <= SCENE[1]:
            rows.append(capture(self, overlay, latest, incoming, hands, t_sec, value))
        return value

    ExchangeLandingProjection._evaluate = evaluate
    dest = OUT/'baseline/review'
    dest.mkdir(parents=True, exist_ok=True)
    replay(Path('logs/e16/records/review.jsonl.gz'), dest,
           Path('models/exchange_event_v3'), True, **OPTIONS)
    save_json(OUT/'BASELINE_EQUIVALENCE.json', compare(Path('logs/e22/on/review'), dest))
    save_json(OUT/'E22_SCENE_TRACE.json', rows)
    flat = [dict(t_sec=r['t_sec'], **{k: v for k, v in s.items()
            if k not in ('board', 'response_board', 'queue')}) for r in rows for s in r['sides']]
    with (OUT/'E22_SCENE_TRACE.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == '__main__':
    main()
