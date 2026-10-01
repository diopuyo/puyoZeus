"""5B固定100局面の全遷移・全値・最善応手を5Cと照合する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache

import numpy as np

from scripts import prefire_v5_experiment as reference
from src import prefire_v5_search as base
from src import prefire_v5_value as value
from src import prefire_v5b_search as exact
from src.prefire_v5b_value import BOARD_CACHE_SIZE, CachedM0, CachedStatic
from src.prefire_v5c_search import TransitionTable, side_options
from src.prefire_v5c_value import EvaluationCache

OUT = Path('logs/prefire_prediction/v5c/verification')


def audit_features(fast: EvaluationCache, original: SimpleNamespace) -> SimpleNamespace:
    """評価に使った全片側特徴を列ごとに比較する。欠測NaN同士は一致とする。"""
    from src.prefire_v5c_value import board_features
    counts = SimpleNamespace(checked=0, mismatches=0)
    @lru_cache(maxsize=BOARD_CACHE_SIZE)
    def checked(raw: bytes, shape: tuple, dtype: str) -> dict:
        actual = board_features(raw, shape, dtype)
        expected = original._build_static.features(raw, shape, dtype)
        counts.checked += 1
        counts.mismatches += not np.array_equal([actual[k] for k in expected],
                                               list(expected.values()), equal_nan=True)
        return actual
    fast.proxy._build_static.features = checked
    return counts


def matrices(states: tuple, row: dict, original: SimpleNamespace, fast: EvaluationCache) -> dict:
    """全22×22遷移を表引き・逐次計算・独立会計で比較する。"""
    expected = tuple(reference.reference_options(p) for p in states)
    actual = tuple(exact.candidates(p, 1, side=i) for i, p in enumerate(states))
    mismatch = sum({p.path: p for p in a} != {p.path: p for p in b} for a, b in zip(expected, actual))
    table, arrays, collisions = TransitionTable(), [], []
    max_error, transitions = 0., 0
    cache: dict[tuple, float] = {}
    for attacker in (0, 1):
        matrix = np.empty((len(expected[0]), len(expected[1])))
        for i, left in enumerate(expected[0]):
            for j, right in enumerate(expected[1]):
                ref = reference.reference_resolve((left, right), attacker, row['elapsed'])
                got = table.resolve(left, right, attacker, row['elapsed'])
                mismatch += ref != got or got != base.resolve(left, right, attacker, row['elapsed'])
                transitions += 1
                key = tuple((p.board, p.queue, p.pending) for p in ref.sides)
                if key not in cache:
                    cache[key] = value.evaluate(original, ref, row['elapsed'])
                matrix[i, j] = cache[key]
                max_error = max(max_error, abs(cache[key]-fast(got, row['elapsed'])))
        arrays.append(matrix)
        collisions.extend(response_dependencies(matrix, expected, attacker, row['elapsed']))
    return dict(transitions=transitions, mismatches=int(mismatch), max_value_error=max_error,
                arrays=arrays, dependencies=collisions)


def response_dependencies(matrix: np.ndarray, options: tuple, attacker: int, elapsed: float) -> list:
    """同じ受け量・手数・連鎖数でも最善応手が変わる実例を保存する。"""
    groups, examples = {}, []
    for index, attack in enumerate(options[attacker]):
        scores = matrix[index, :] if attacker == 0 else matrix[:, index]
        selected = int(np.argmin(scores) if attacker == 0 else np.argmax(scores))
        key = (int(base.sim.send_ojama(attack.score, elapsed)), attack.consumed, attack.chains)
        previous = groups.get(key)
        if previous is not None and selected != previous[1]:
            old_scores = matrix[previous[0], :] if attacker == 0 else matrix[:, previous[0]]
            gap = abs(float(scores[previous[1]]-scores[selected]))
            if gap > reference.VALUE_TOLERANCE and abs(float(old_scores[selected]-old_scores[previous[1]])) > reference.VALUE_TOLERANCE:
                examples.append(dict(attacker=attacker, key=key, attacks=[previous[0], index],
                                     responses=[previous[1], selected], value_gap=gap))
        groups[key] = (index, selected)
    return examples[:2]


def check(row: dict, original: SimpleNamespace, fast: EvaluationCache) -> dict:
    """有限一手の最善値と5Bの選択手順まで比較する。"""
    states = tuple(base.Position(bytes(g), tuple(k), p)
                   for g, k, p in zip(row['boards'], row['known'], row['pending']))
    result = matrices(states, row, original, fast)
    arrays = result.pop('arrays')
    table, choices = TransitionTable(), []
    for attacker in (0, 1):
        matrix = arrays[attacker]
        expected = float(matrix.min(axis=1).max() if attacker == 0 else matrix.max(axis=0).min())
        sequential = exact.choose(states, attacker, row['elapsed'], fast, depth=1)
        found = exact.choose(states, attacker, row['elapsed'], fast, depth=1,
                             resolver=table.resolve, options_fn=side_options)
        choices.append(dict(exact=expected, actual=found[0],
            missed=abs(expected-found[0]) > reference.VALUE_TOLERANCE,
            table_sequential_equal=found == sequential))
    return dict(source=row['source'], t_sec=row['t_sec'], **result, values=choices)


def main() -> None:
    """保存済み原票を上書きせず、範囲を指定して再開できる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=100)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    overlay = reference.model_overlay()
    original = SimpleNamespace(**{**vars(overlay), '_m0': CachedM0(overlay._m0),
                                   '_build_static': CachedStatic(overlay._build_static)})
    fast = EvaluationCache(overlay)
    counts = audit_features(fast, original)
    samples = json.loads((reference.OUT/'samples_ledger.json').read_text())
    for index in range(args.start, args.stop):
        path = args.out/f'case_{index:03d}.json'
        if path.exists():
            continue
        before = (counts.checked, counts.mismatches)
        result = check(samples[index], original, fast)
        result.update(feature_boards=counts.checked-before[0],
                      feature_mismatches=counts.mismatches-before[1])
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(index, result, flush=True)


if __name__ == '__main__':
    main()
