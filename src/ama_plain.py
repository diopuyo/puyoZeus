"""amaの最善初手と代表計画から期待得点を求める独立試作。本番既定OFF。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter
from src.ama_plain_reference import Reference, DEPTH
from src.ama_rules_v5 import landing_proof, DEATH_CELL
from src.b3_timing import TimingTable
from src.exchange_count_ledger import gross
from src.prefire_race_physics import hand
from src.prefire_worker import validate, CELLS


def validate_request(request: object) -> None:
    """死亡セルだけは連鎖後に判定する。その他の入力契約は既存検査を保つ。"""
    boards = []
    for original in request.boards:
        board = list(original)
        if len(board) == CELLS and board[DEATH_CELL] in (0, 9, *request.colors):
            board[DEATH_CELL] = 0
        boards.append(tuple(board))
    validate(replace(request, boards=tuple(boards)))


def capacity(board: list) -> int:
    """無連鎖で左列から降る既存規則の、死亡セルに達する直前の受入個数。"""
    count = 0
    while count < len(board):
        # 1個ずつの呼出は毎回左列からになるため、最大30個ずつ同じ原盤へ試す。
        batch = min(30, len(board)-count)
        landed = 0
        for n in range(1, batch+1):
            proof = landing_proof(board, n, 'carry')
            if proof['dead']:
                return count+landed
            landed = proof['landed']
        if proof['landed'] == 0:
            return count
        board = proof['board']
        count += landed
    return count


def initial_state(board: list, result: dict, carry: int, elapsed: float,
                  all_clear: bool = False) -> tuple:
    """開始盤の自動連鎖を配置計画と分け、連鎖後の端数・時刻を引き継ぐ。"""
    initial, table = result.get('initial', {}), TimingTable()
    scores = initial.get('scores', [])
    duration = len(scores)*table.stage_sec+table.tail_sec if scores else 0.
    sent, remainder = gross(scores, [elapsed+(i+1)*table.stage_sec
                                    for i in range(len(scores))], carry, 0., all_clear)
    after = initial.get('board', board)
    automatic = dict(h=0, chain=len(scores), sent=sent, scores=scores,
                     duration_sec=duration, board=after, active=bool(scores))
    return after, remainder, elapsed+duration, automatic


def plan(board: list, result: dict, carry: int, elapsed: float, all_clear: bool = False) -> dict:
    board, carry, elapsed, automatic = initial_state(board, result, carry, elapsed, all_clear)
    following_clear = not any(board) if automatic['active'] else all_clear
    terminal = bool(board[DEATH_CELL])
    value = planned_fire(board, dict(result, candidates=[]) if terminal else result,
                         carry, elapsed, following_clear)
    return dict(value, automatic_fire=automatic, read_board=board, terminal=terminal)


def planned_fire(board: list, result: dict, carry: int, elapsed: float,
                 all_clear: bool = False) -> dict:
    candidates, table = result['candidates'], TimingTable()
    if not candidates:
        return dict(h=None, chain=0, sent=0, consumption=0., ama_score=0,
            immediate=False, no_safe_move=True, capacity=capacity(board), steps=[])
    best = candidates[0]
    before, clock, fired = list(board), 0., None
    for h, step in enumerate(best['steps'], 1):
        clock += hand(before, table)
        if step['scores']:
            sent, _ = gross(step['scores'],
                [elapsed+clock+(i+1)*table.stage_sec for i in range(len(step['scores']))],
                carry, 0., all_clear)
            material = sum(c not in (0, 9) for c in before)+2
            consumed = material-sum(c not in (0, 9) for c in step['board'])
            fired = dict(h=h, chain=len(step['scores']), sent=sent,
                         consumption=consumed/max(1, material))
            break
        before = step['board']
    return dict(fired or dict(h=None, chain=0, sent=0, consumption=0.),
        ama_score=best['score'], immediate=bool(best['root_score']), no_safe_move=False,
        capacity=capacity(board), representative_branch=best['representative_branch'],
        placement=[best['x'], best['r']], steps=best['steps'], branches=best['branches'])


def calculate(request: object, carries: list, model: dict | None = None,
              enabled: bool = False, references: list | None = None, parallel: bool = False,
              tapes: list | None = None, all_clear: list | None = None) -> dict | None:
    if not enabled:
        return None
    began = perf_counter()
    validate_request(request)
    if len(carries) != 2 or any(c is None or c < 0 for c in carries):
        raise ValueError('保存端数が必要')
    refs = references or [Reference(), Reference()]
    try:
        clear_flags = all_clear if all_clear is not None else [False, False]
        if len(clear_flags) != 2:
            raise ValueError('全消しフラグは両側分が必要')
        visible = [t[:3] for t in tapes] if tapes is not None else request.known
        jobs = [(r, list(b), list(v), request.colors) for r, b, v in
                zip(refs, request.boards, visible)]
        reading = perf_counter()
        if parallel:
            with ThreadPoolExecutor(2) as pool:
                reads = list(pool.map(lambda args: args[0].think(*args[1:]), jobs))
        else:
            reads = [r.think(b, v, c) for r, b, v, c in jobs]
        converted = perf_counter()
        plans = [plan(list(b), r, c, request.elapsed_sec, clear) for b, r, c, clear in
                 zip(request.boards, reads, carries, clear_flags)]
        value = dict(plans=plans, pending=list(request.pending))
        if model is not None:
            from src.ama_plain_probability import probability
            value['probability'] = probability(value, model)
        return dict(value, wall_sec=perf_counter()-began,
            setup_sec=reading-began, read_pair_wall_sec=converted-reading,
            conversion_sec=perf_counter()-converted, read_wall_sec=[r['wall_sec'] for r in reads],
            read_cpu_sec=[r['cpu_sec'] for r in reads], reused=[r['reused'] for r in reads],
            read_stages=[{k: r[k] for k in ('search_wall_sec', 'path_wall_sec', 'native_wall_sec')
                          if k in r} for r in reads],
            reference_calls=sum(not r['reused'] for r in reads))
    finally:
        if references is None:
            for ref in refs:
                ref.close()
