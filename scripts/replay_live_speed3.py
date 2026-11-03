"""FIX4と同じ全通知再投入で探索費用と再利用余地を観測する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import hashlib
import inspect
import json
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(_public_dev_text('<work>/wt_live_speed'))
OUT = Path(_public_dev_text('<work>/speed3'))
CAPACITY = 2048
sys.path[:0] = [str(ROOT), _public_dev_text('<work>/fix4')]
import replay_reference as base
from src import indicators_v2 as iv
from src import exchange_prefire_candidates as prefire
from src.exchange_event_record import encode
from src.scoring import compute_effective_rate

CALLS: list[dict] = []
UPDATES: list[dict] = []
SEEN: dict[str, set] = {}
RECENT: dict[str, OrderedDict] = {}
INPUTS: set[bytes] = set()
CONTEXT: dict = {}
INPUT_DUPLICATES = 0


def observe(function: Any, name: str) -> Any:
    signature = inspect.signature(function)

    def measured(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        board = values.pop('board')
        simulator = values.pop('sim', values.pop('simulator', None))
        settings = (type(simulator).__module__, type(simulator).__qualname__,
                    getattr(simulator, '_exclude_hidden_row_from_pop', None),
                    getattr(simulator, '_fast_groups', None))
        if 'elapsed_sec' in values:
            values['elapsed_sec'] = compute_effective_rate(values['elapsed_sec'])
        key = pickle.dumps((board._grid.dtype.str, board._grid.shape,
                            board._grid.tobytes(), settings, values))
        seen, recent = SEEN.setdefault(name, set()), RECENT.setdefault(name, OrderedDict())
        repeated, hit = key in seen, key in recent
        started = perf_counter()
        result = function(*args, **kwargs)
        elapsed = (perf_counter()-started)*1000
        CALLS.append(dict(name=name, ms=elapsed, repeated=repeated, hit=hit, **CONTEXT))
        seen.add(key)
        recent[key] = True
        recent.move_to_end(key)
        if len(recent) > CAPACITY:
            recent.popitem(last=False)
        return result
    return measured


class MeasuredOverlay(base.MeasuredOverlay):
    def update(self, *args: Any, **kwargs: Any) -> None:
        global INPUT_DUPLICATES
        result, snapshot = args[:2]
        sides = (result.p1, result.p2)
        states = tuple(side.state.name for side in sides)
        group = 'chain' if 'CHAIN' in states else 'ojama' if 'OJAMA_FALL' in states else 'other'
        CONTEXT.update(t_sec=args[3], group=group)
        values = [(s.confirmed_board, s.next_pair, s.dnext_pair) for s in sides]
        key = hashlib.sha256(pickle.dumps(encode((values, snapshot)))).digest()
        INPUT_DUPLICATES += key in INPUTS
        INPUTS.add(key)
        first = len(CALLS)
        super().update(*args, **kwargs)
        row = base.ROWS[-1]
        UPDATES.append(dict(t_sec=args[3], group=group, update_ms=row['update_ms'],
                            reference_ms=row['after_ms'], search_ms=sum(c['ms'] for c in CALLS[first:])))


def main() -> None:
    os.environ['PUYO_LIVE_REFERENCE_TIMES_ONLY'] = '1'
    base.OUT, base.MODE, base.PROFILE = OUT, '1', False
    base.MeasuredOverlay = MeasuredOverlay
    iv._near_future_search = observe(iv._near_future_search, 'near_future')
    prefire.enumerate_candidates = observe(prefire.enumerate_candidates, 'prefire_candidates')
    try:
        base.main()
    finally:
        value = dict(status='diagnostic_not_quality_pass', updates=UPDATES, calls=CALLS,
                     input_n=len(UPDATES), input_duplicates=INPUT_DUPLICATES,
                     input_unique=len(INPUTS), capacity=CAPACITY)
        (OUT/'replay_diagnostic.json').write_text(json.dumps(value), encoding='utf-8')
        print('探索呼出', len(CALLS), '入力重複', INPUT_DUPLICATES, '/', len(UPDATES), flush=True)


if __name__ == '__main__':
    main()
