"""先回り試作の裏計算。到着済みの確定盤だけを受け取り、本番からは呼ばない。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src import prefire_work_budget as work_budget

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.b3_timing import TimingTable
from src.phase_j._live_ex_frozen.src.exchange_projection import hand_secs, hands_for
from src.phase_j._live_ex_frozen.src.scoring import compute_effective_rate
from src.phase_j._live_ex_frozen.src.production_config import PREFIRE_V4_ENABLED_DEFAULT

LOW_SPEC_RATIO = 1.6473854230
MAX_LOW_SPEC_SEC = .5
VISIBLE_ATTACK_HANDS = 2
ROLLOUTS = 4
WIDTH, DEPTH = 1, 2
CELLS, COLS, DEATH_CELL = 78, 6, 8
MAX_HANDS = 128
MS = 1000.
RETURN_MARGIN_SEC = .04


@dataclass(frozen=True)
class PrefireInput:
    """未来の勝敗・実発火側・未配送ツモを受け取る欄は持たない。"""
    game: int
    generation: int
    input_sec: float
    evidence_times: tuple[float, ...]
    boards: tuple[tuple[int, ...], tuple[int, ...]]
    known: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    colors: tuple[int, ...]
    pending: tuple[int, int]
    elapsed_sec: float
    stable: tuple[bool, bool]
    seed: int


def validate(request: PrefireInput) -> None:
    if not np.isfinite(request.input_sec) or not request.evidence_times:
        raise ValueError('入力時刻または証跡がありません')
    if any(not np.isfinite(t) or t > request.input_sec for t in request.evidence_times):
        raise ValueError('未来の入力')
    if not all(request.stable):
        raise ValueError('両者STABLEの確定盤が必要です')
    if len(set(request.colors)) != 4 or any(c not in range(1, 6) for c in request.colors):
        raise ValueError('観測4色が未確定です')
    for board, known in zip(request.boards, request.known):
        if len(board) != CELLS or any(c not in (0, 9, *request.colors) for c in board):
            raise ValueError('不明色または不正盤面')
        if board[DEATH_CELL] or len(known) < VISIBLE_ATTACK_HANDS:
            raise ValueError('死亡盤または既知ツモ不足')
        if any(len(p) != 2 or any(c not in request.colors for c in p) for p in known):
            raise ValueError('既知ツモの色矛盾')
    if any(p < 0 for p in request.pending):
        raise ValueError('負の予告')


def maximum_attack(board: tuple[int, ...], known: tuple, native: Any) -> dict:
    """0手と既知2手の全合法配置。発火済み経路を次手へ延ばさない。"""
    initial = native.simulate_chain_py(list(board), True)
    best = dict(chain=initial.chain_count, score=initial.exact_score, hands=0,
                final_board=list(initial.final_grid))
    if initial.chain_count:
        return best
    frontier = [list(board)]
    for hand, pair in enumerate(known[:VISIBLE_ATTACK_HANDS], 1):
        following = {}
        for current in frontier:
            for result in native.enumerate_and_simulate_placements_py(current, *pair, False, True):
                if result.final_grid[DEATH_CELL]:
                    continue
                if result.chain_count:
                    candidate = dict(chain=result.chain_count, score=result.exact_score,
                                     hands=hand, final_board=list(result.final_grid))
                    if (candidate['chain'], candidate['score'], -hand) > (
                            best['chain'], best['score'], -best['hands']):
                        best = candidate
                else:
                    following[tuple(result.final_grid)] = result.final_grid
        frontier = list(following.values())
    return best


def rollout(task: tuple, native: Any) -> dict:
    board, pairs, k, colors, target, deadline = task
    # 方式Fは3組までの可視先読み。未知列を知った全手先読みにはしない。
    value = native.ama_f_py(list(board), pairs, k, WIDTH, DEPTH, list(colors),
        policy='main', incoming=0, target_score=target,
        budget_ms=max(0., deadline-perf_counter())*MS if deadline is not None else None)
    return dict(score=value['max_score'], coverage=value['coverage'],
                fire_hand=value['fire_hand'], dead=value['dead'], truncated=value.get('truncated', False))


def return_capacity(board: tuple, known: tuple, colors: tuple, k: int, attack: float,
                    rate: float, seed: int, native: Any, pool: ThreadPoolExecutor,
                    deadline: float | None = None) -> dict:
    if attack <= 0:
        return dict(amount=0., samples=[], quick=0., k=k)
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(ROLLOUTS):
        unknown = rng.choice(colors, size=(max(0, k+DEPTH-len(known)), 2))
        pairs = [tuple(p) for p in known]+[tuple(map(int, p)) for p in unknown]
        tasks.append((board, pairs, k, colors, round(attack*rate), deadline))
    values = list(pool.map(lambda task: rollout(task, native), tasks))
    if any(v['truncated'] for v in values):
        raise TimeoutError('Fロールアウト期限。未完の返却を0個として扱わない')
    quick = maximum_attack(board, known[:min(k, VISIBLE_ATTACK_HANDS)], native)
    quantities = [max(quick['score'], v['score'])//rate for v in values]
    return dict(amount=float(np.mean(quantities)), samples=quantities,
                quick=float(quick['score']//rate), k=k)


def calculate(request: PrefireInput, native: Any, table: TimingTable,
              pool: ThreadPoolExecutor, enabled: bool = PREFIRE_V4_ENABLED_DEFAULT) -> dict | None:
    """live裏ジョブ用入口。既定OFF。時間超過も実測値のまま返し、公開側で拒否する。"""
    if not enabled:
        return None
    began = perf_counter()
    validate(request)
    try:
        threats = calculate_sides(request, native, table, pool, began)
        payload = dict(sides=threats, x=threats[0]['x']-threats[1]['x'])
    except TimeoutError as error:
        payload = dict(reason=str(error), budget_exhausted=True)
    seconds = perf_counter()-began
    return dict(game=request.game, generation=request.generation, input_sec=request.input_sec,
        evidence_times=list(request.evidence_times), elapsed_ms=seconds*MS,
        low_spec_ms=seconds*MS*LOW_SPEC_RATIO, ready_sec=request.input_sec+seconds*LOW_SPEC_RATIO,
        over_budget=False if work_budget.enabled() else seconds*LOW_SPEC_RATIO > MAX_LOW_SPEC_SEC, **payload)


def calculate_sides(request: PrefireInput, native: Any, table: TimingTable,
                    pool: ThreadPoolExecutor, began: float) -> list[dict]:
    deadline = None if work_budget.enabled() else began+(MAX_LOW_SPEC_SEC-RETURN_MARGIN_SEC)/LOW_SPEC_RATIO
    rate = compute_effective_rate(max(0., request.elapsed_sec), from_first_move=True)
    attacks = [maximum_attack(b, q, native) for b, q in zip(request.boards, request.known)]
    threats = []
    for side, attack in enumerate(attacks):
        amount = max(0., attack['score']//rate-request.pending[side])
        duration = table.stage_sec*attack['chain']+table.tail_sec if attack['chain'] else 0.
        hand = hand_secs(table, np.array(request.boards[1-side]).reshape(-1, COLS))[0]
        k = min(MAX_HANDS, hands_for(duration, hand))
        returned = return_capacity(request.boards[1-side], request.known[1-side], request.colors,
            k, amount, rate, request.seed, native, pool, deadline)
        returned['gross_amount'] = returned['amount']
        returned['amount'] = max(0., returned['amount']-request.pending[1-side])
        threats.append(dict(attack=amount, returned=returned['amount'], x=amount-returned['amount'],
            duration=duration, chain=attack['chain'], hands=attack['hands'], response=returned))
    return threats
