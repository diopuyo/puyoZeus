"""原Pipeline/SM/trackerに人工の消去前後を渡し、入口と出口を観測する。実J合格ではない。"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import pytest
import test_tracker_input as F
import tracker_input as T

ROOT = Path(__file__).resolve().parent
FIRST, STRIDE, PREPARE, LIMIT = 34772, 2, 40, 240


def drive(real: Any, patch: Any) -> dict[str, Any]:
    pipe, _, _, image, _ = real
    module = sys.modules[type(pipe).__module__]
    source = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v36/PROBABILISTIC_BASIS.jsonl'
    packet = json.loads(source.read_bytes())['state']
    before = module.Board.from_dict({'grid': packet['hidden_worlds'][0]['cells'] + packet['visible']})
    outcome = pipe._chain_tracker_1p._simulator.simulate(before)
    current, other = {'board': before}, module.Board()
    patch.setattr(pipe._reader, 'read_both_boards', lambda *_args, **_kwargs: (current['board'].copy(), other.copy()))
    rows, seen, failure = [], False, None
    for index in range(LIMIT):
        frame, clock = FIRST + index * STRIDE, (FIRST + index * STRIDE) / 60
        if index == PREPARE: current['board'] = outcome.final_board
        try:
            pipe.update(frame, clock, image)
        except Exception as error:
            failure = dict(frame=frame, error=repr(error))
            break
        origin = pipe._active_chain_1p
        seen = seen or origin is not None
        row = T.observe(pipe) | dict(frame=frame, index=index, state=pipe._sm_1p.context.state.value,
            confirmed_count=None if pipe._sm_1p.context.confirmed_board is None else
                int((pipe._sm_1p.context.confirmed_board._grid != 0).sum()),
            raw_count=int((current['board']._grid != 0).sum()),
            chain_banned=clock - pipe._match_active_started_time < pipe.CHAIN_BAN_SEC_AFTER_MATCH_START,
            trigger=None if origin is None else origin.trigger_sec,
            origin_before_matches=None if origin is None else origin.before_board.to_dict() == before.to_dict())
        rows.append(row)
        if seen and origin is None and row['state'] == 'stable': break
    return dict(rows=rows, error=failure, source=str(source), single_world_artificial_input=True,
        simulated_chain_count=outcome.chain_count, origin_seen=seen,
        origin_closed=seen and pipe._active_chain_1p is None, actual_J=False, quality_gate_clear=False)


def main() -> int:
    started, evidence = time.monotonic(), {}
    with contextlib.contextmanager(F.F.frozen.__wrapped__)() as frozen:
        with pytest.MonkeyPatch.context() as patch:
            supplied = T.transport(F.F.real.__wrapped__, evidence)
            with contextlib.contextmanager(supplied)(frozen, patch) as real:
                report = drive(real, patch)
    report.update(constructor=evidence, seconds=time.monotonic() - started)
    with (ROOT / 'TRACKER_CASCADE_PROBE_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('origin_seen', 'origin_closed', 'simulated_chain_count', 'seconds')}))
    return int(report['error'] is not None)


if __name__ == '__main__':
    raise SystemExit(main())
