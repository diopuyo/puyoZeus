"""実験1: 固定した実局面でPython参照遷移と短範囲完全探索を比較する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src import prefire_v5_search as search
from src import prefire_v5_value as value
from src.board import Board
from src.board import BOARD_COLS
from src.chain import ChainSimulator
from src.exchange_event_record import read_records
from src.indicators_v2 import _place_pair_to_board
from src.prefire_stable_queue import SideQueue
from src.scoring import calculate_chain_score, OJAMA_MAX_DROP_PER_TURN
from scripts.run_prefire_replay_20260930 import RECORDS, BASELINE_DIRS, model_directory, options

OUT = Path('logs/prefire_prediction/v5_experiment')
SAMPLE_COUNT = 100
REFERENCE_DEPTH = 1  # 短い完全探索: 両者1組、最大22×22通り。NEXT/NEXT2も固定。
ROTATIONS = range(4)
SEED = 20261001
VALUE_TOLERANCE = 1e-10


def collect() -> list[dict]:
    """両側STABLE・6色既知の重複なし入力を採取し、母集団から固定seedで選ぶ。"""
    pool, seen = [], set()
    for source in BASELINE_DIRS:
        queues = [SideQueue(), SideQueue()]
        last_game, start = None, 0.0
        for row in read_records(RECORDS / f'{source}.jsonl.gz'):
            if row['kind'] != 'update':
                continue
            result, snapshot, _, stamp, game = row['args'][:5]
            if game != last_game:
                queues, last_game, start = [SideQueue(), SideQueue()], game, stamp
            grids = []
            for idx, side in enumerate((result.p1, result.p2)):
                if side.state.value != 'stable' or side.confirmed_board is None:
                    break
                board = side.confirmed_board
                grid = board._grid if isinstance(board, Board) else np.asarray(board)
                grids.append(grid.astype(np.int8).tobytes())
                queues[idx].observe(stamp, grids[-1], np.array([*(side.next_pair or (0, 0)),
                                                               *(side.dnext_pair or (0, 0))]))
            if len(grids) != 2:
                continue
            known = tuple(q.known() for q in queues)
            key = (tuple(grids), known)
            if key in seen or any(len(k) != 6 or any(c not in search.sim.PLAYABLE_COLORS for c in k) for k in known):
                continue
            if any(search.status(g, k) != search.OK for g, k in zip(grids, known)):
                continue
            # 保存記録は両側総量がないため、純差からの復元を行わず検証用0で固定する。
            seen.add(key)
            pool.append(dict(source=source, t_sec=stamp, game=game, elapsed=stamp-start,
                boards=[list(g) for g in grids], known=known, pending=(0, 0),
                pending_provenance='controlled_zero_not_observed'))
    rng = np.random.default_rng(SEED)
    indices = sorted(rng.choice(len(pool), min(SAMPLE_COUNT, len(pool)), replace=False).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'population.json').write_text(json.dumps(dict(eligible=len(pool), selected=indices)), encoding='utf-8')
    selected = [pool[i] for i in indices]
    (OUT/'samples.json').write_text(json.dumps(selected), encoding='utf-8')
    return selected


def reference_options(state: search.Position) -> tuple[search.Position, ...]:
    """独立したPython配置・連鎖・公式得点計算で1手を全列挙する。"""
    simulator = ChainSimulator(exclude_hidden_row_from_pop=search.GHOST_CHAIN_RULE_ENABLED)
    out = []
    for rotation in ROTATIONS:
        for col in range(BOARD_COLS):
            board = _place_pair_to_board(search.sim._board(state.board), state.queue[:2], col, rotation)
            if board is None:
                continue
            chain = simulator.simulate(board)
            out.append(search.Position(chain.final_board._grid.astype(np.int8).tobytes(), state.queue[2:],
                state.pending, calculate_chain_score(chain).total_score, 1, chain.chain_count, ((col, rotation),)))
    return tuple(out)


def reference_resolve(sides: tuple, attacker: int, elapsed: float) -> search.Exchange:
    """会計を独立計算し、同じ決定論的着弾規則の下で盤面と残量を照合する。"""
    from dataclasses import replace
    pending, cancelled = [p.pending for p in sides], [0, 0]
    for side in (attacker, 1-attacker):
        sent = int(search.sim.send_ojama(sides[side].score, elapsed))
        cancelled[side] = min(sent, pending[side])
        pending[side] -= cancelled[side]
        pending[1-side] += sent-cancelled[side]
    boards = tuple(search.sim._board(p.board) for p in sides)
    landed = tuple(search.land_pending_ojama_onto_board(b, boards[1-i], pending[i])[0]
                   for i, b in enumerate(boards))
    dropped = tuple(min(v, OJAMA_MAX_DROP_PER_TURN) for v in pending)
    after = tuple(replace(p, board=landed[i]._grid.astype(np.int8).tobytes(), pending=pending[i]-dropped[i])
                  for i, p in enumerate(sides))
    return search.Exchange(after, tuple(cancelled), dropped, tuple(b.is_dead() for b in landed))


def model_overlay() -> SimpleNamespace:
    """再生と同じ保存モデル・G_fe変換をロードする。"""
    from src.exchange_event_evaluator import FileExchangeModels
    from src.exchange_event_m0 import FileM0Predictor
    from scripts.visualize_advantage_overlay import _exchange_static_input
    directory = model_directory(options('v5'))
    return SimpleNamespace(_m0=FileM0Predictor(directory/'M0'),
        _build_static=_exchange_static_input, _snapshots=[(0, SimpleNamespace())],
        tracker=SimpleNamespace(models=FileExchangeModels.load(directory, lightweight=True)))


def check_sample(row: dict, overlay: SimpleNamespace) -> dict:
    """同じ手順の遷移全件と、K制限による最善値の損失を別々に数える。"""
    states = tuple(search.Position(bytes(g), tuple(k), p) for g, k, p in zip(row['boards'], row['known'], row['pending']))
    expected = tuple(reference_options(p) for p in states)
    actual = tuple(search.candidates(p, REFERENCE_DEPTH, None) for p in states)
    mismatch = sum(dict((p.path, p) for p in a) != dict((p.path, p) for p in b) for a, b in zip(actual, expected))
    cache: dict = {}
    def evaluator(exchange: search.Exchange, elapsed: float) -> float:
        key = (exchange, elapsed)
        if key not in cache:
            cache[key] = value.evaluate(overlay, exchange, elapsed)
        return cache[key]
    transition_count, exact_values = 0, []
    for attacker in (0, 1):
        matrix = []
        for left in expected[0]:
            values = []
            for right in expected[1]:
                ref = reference_resolve((left, right), attacker, row['elapsed'])
                got = search.resolve(left, right, attacker, row['elapsed'])
                mismatch += ref != got
                transition_count += 1
                values.append(evaluator(ref, row['elapsed']))
            matrix.append(values)
        array = np.asarray(matrix)
        exact_values.append(float(array.min(axis=1).max() if attacker == 0 else array.max(axis=0).min()))
    results = []
    for attacker in (0, 1):
        limited = value.choose(states, attacker, row['elapsed'], evaluator, REFERENCE_DEPTH, search.TOP_K)
        exact = exact_values[attacker]
        results.append(dict(exact=exact, limited=limited[0], missed=abs(exact-limited[0]) > VALUE_TOLERANCE))
    return dict(source=row['source'], t_sec=row['t_sec'], transitions=transition_count,
                mismatches=mismatch, values=results)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--limit', type=int, default=SAMPLE_COUNT)
    parser.add_argument('--ledger', action='store_true')
    args = parser.parse_args()
    samples = collect() if args.collect else json.loads((OUT/('samples_ledger.json' if args.ledger else 'samples.json')).read_text())
    if args.ledger:
        OUT = OUT/'ledger'
        OUT.mkdir(parents=True, exist_ok=True)
    overlay, results = model_overlay(), []
    for i, row in enumerate(samples[:args.limit]):
        result = check_sample(row, overlay)
        results.append(result)
        (OUT/f'case_{i:03d}.json').write_text(json.dumps(result), encoding='utf-8')
        print(i, result, flush=True)
    summary = dict(samples=len(results), transitions=sum(r['transitions'] for r in results),
        mismatches=sum(r['mismatches'] for r in results),
        missed=sum(any(v['missed'] for v in r['values']) for r in results))
    (OUT/'summary.json').write_text(json.dumps(summary), encoding='utf-8')


if __name__ == '__main__':
    main()
