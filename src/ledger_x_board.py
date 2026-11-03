"""台帳を必要としない発火盤面の火力。勝率・正味を補完しない。"""
from __future__ import annotations

from typing import Any

from src.exchange_count_ledger import FORMULA_SCORE_DELAY_SEC, STEP_SEC, gross
from src.exchange_count_prediction import simulate
from src.scoring import OJAMA_RATE_STANDARD


def generated_range(scores: list[int], trigger: float, ledger: dict) -> dict:
    """連鎖得点由来の個数だけを換算する。全消し予約は含まない。"""
    carry, first = ledger.get('carry'), ledger.get('first_move')
    if not scores:
        return dict(low=0, high=0, exact=0)
    carries = range(OJAMA_RATE_STANDARD) if carry is None else [carry]
    if first is None:
        low = (sum(scores) + min(carries)) // OJAMA_RATE_STANDARD
        high = sum(scores) + max(carries)
    else:
        times = [trigger + FORMULA_SCORE_DELAY_SEC + STEP_SEC * i for i in range(len(scores))]
        values = [gross(scores, times, value, first)[0] for value in carries]
        low, high = min(values), max(values)
    return dict(low=low, high=high, exact=low if low == high else None)


def side_power(source: dict, chain: dict | None, ledger: dict, native: Any,
               game_start: float | None = None) -> dict | None:
    """既存候補があれば使い、初段式が未到達でも発火前盤面を計算する。"""
    event = (source.get('chain_event') or {}).get('namespace')
    if not event:
        return None
    trigger = event['trigger_sec']
    if game_start is not None and trigger < game_start:
        return None
    if chain and chain['signature'][0] == trigger:
        scores, method = list(chain['best']['scores']), '既存盤面候補'
    else:
        board = (event.get('before_board') or {}).get('board', {}).get('array')
        if board is None:
            return None
        flat = [cell for row in board for cell in row]
        if any(cell not in (0, 1, 2, 3, 4, 5, 9) for cell in flat):
            return None
        scores, method = simulate(flat, native, compact=True)['scores'], '発火前盤面'
    count = len(scores)
    return dict(trigger=trigger, chain_count=count, score_points=sum(scores), step_scores=scores,
                first_to_last_step_sec=STEP_SEC * max(count - 1, 0),
                last_score_delay_sec=FORMULA_SCORE_DELAY_SEC + STEP_SEC * (count - 1) if count else None,
                generated_cells=generated_range(scores, trigger, ledger),
                carry=ledger.get('carry'), first_move=ledger.get('first_move'),
                net_cells=None, all_clear_included=False, method=method)


def board_record(stamp: float, game: int, raw: list[dict], chains: list,
                 ledgers: list[dict] | None, native: Any,
                 game_start: float | None = None) -> dict | None:
    """正味が不明でも盤面由来の計算結果を公開可能な票にする。"""
    sides = [side_power(source, chains[i], (ledgers or [{}, {}])[i], native, game_start)
             for i, source in enumerate(raw)]
    if not any(side is not None for side in sides):
        return None
    return dict(game=game, input_sec=stamp, board_only=True, board_firepower=sides,
                predicted_scores=[side['step_scores'] if side else [] for side in sides],
                active=False, death_sides=[], evidence_times=[stamp],
                net_cells=None, missing='正味未確定（盤面火力のみ）')
