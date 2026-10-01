"""5B修正前のK枝落ちを、5Aと同じ100入力で保存する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import prefire_v5_experiment as experiment
from src import prefire_v5_search as search
from src import prefire_v5_value as value

OUT = Path('logs/prefire_prediction/v5b/before')
LEGACY_K = 8


def describe(candidate: search.Position, options: tuple) -> dict:
    """同じ手数・発火群での得点順位と、枝落ちした配置を記録する。"""
    group = sorted((p for p in options if (p.consumed, bool(p.chains)) ==
                    (candidate.consumed, bool(candidate.chains))), key=lambda p: (-p.score, p.path))
    rank = group.index(candidate) + 1
    return dict(path=candidate.path, score=candidate.score, consumed=candidate.consumed,
                rank=rank, dropped=rank > LEGACY_K, board=list(candidate.board), queue=candidate.queue)


def run(index: int, overlay: object | None = None) -> None:
    """参照行列を既存照合器で作り、同じ評価キャッシュから枝落ち内訳を採取する。"""
    samples = json.loads((experiment.OUT/'samples_ledger.json').read_text())
    original, details = value.choose, []

    def audited(states: tuple, attacker: int, elapsed: float, evaluator: value.ValueFunction,
                depth: int = 1, k: int | None = LEGACY_K) -> tuple | None:
        limited = original(states, attacker, elapsed, evaluator, depth, LEGACY_K)
        exact = original(states, attacker, elapsed, evaluator, depth, None)
        options = tuple(search.candidates(p, depth, None) for p in states)
        details.append(dict(attacker=attacker, exact_value=exact[0], limited_value=limited[0],
            exact_attack=describe(exact[1], options[attacker]),
            exact_response=describe(exact[2], options[1-attacker]),
            limited_attack=describe(limited[1], options[attacker]),
            limited_response=describe(limited[2], options[1-attacker])))
        return limited

    value.choose = audited
    try:
        result = experiment.check_sample(samples[index], overlay or experiment.model_overlay())
    finally:
        value.choose = original
    result['branches'] = details
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'case_{index:03d}.json').write_text(json.dumps(result), encoding='utf-8')
    print(index, result['values'], flush=True)


def main() -> None:
    """再開時は保存済みの原票を再計算しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=100)
    args = parser.parse_args()
    overlay = experiment.model_overlay()
    for index in range(args.start, args.stop):
        if not (OUT/f'case_{index:03d}.json').exists():
            run(index, overlay)


if __name__ == '__main__':
    main()
