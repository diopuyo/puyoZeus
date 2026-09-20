"""終端後のproducer/consumer照合。未来時刻とACKは明示的な人工対照。"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import importlib.util
import sys
REPAIR = Path(__file__).resolve().parent
BASE = REPAIR.parent / 'g2_second_terminal_arrival_2026-09-14_v1'
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('_raw_terminal_consumer', REPAIR/'terminal_saved.py')
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

import test_saved_boundary as S
from test_terminal_registry import setup
parts = S.parts


def produce(parts: Any, tmp_path: Path, late_ack: bool) -> tuple:
    terminal, binding, _, _ = setup(parts, tmp_path)
    terminal.commit()
    start, ledger, mode = binding.ledger.clock, binding.ledger, binding.mode
    observed, _ = mode.stable(None, None)
    arrival = ledger.arrivals[-1]
    pending = []
    if late_ack:
        binding.ledger = binding.L.acknowledge(ledger, arrival.token, arrival.pair, start + 2, 'synthetic-late')
        event = parts.mode.CORE.V1.N.Consumption('synthetic-late', arrival.token, arrival.pair, start + 2)
        mode.native.pending.append(event)
        pending.append(event)
    else:
        binding.ledger = binding.L.advance_clock(ledger, start + 2)
    first_ledger = binding.ledger
    terminal.capture(dict(events=[]))
    terminal.progress(dict(scope=dict(frame_idx=start + 2), token='synthetic-stable'), None, {})
    binding.ledger = binding.L.advance_clock(binding.ledger, start + 4)
    mode.stable = lambda item, result: (None, 'inactive_or_effect_window')
    terminal.capture(dict(events=[]))
    terminal.progress(dict(scope=dict(frame_idx=start + 4), token='synthetic-inactive'), None, {})
    terminal.close()
    records = [json.loads(line) for line in (tmp_path / 'terminal.jsonl').read_text().splitlines()]
    return terminal, binding, first_ledger, pending, records, observed


@pytest.mark.parametrize('late_ack', [False, True])
def test_saved_tail_matches_actual_producer(parts: Any, tmp_path: Path, late_ack: bool) -> None:
    terminal, binding, first_ledger, pending, records, observed = produce(parts, tmp_path, late_ack)
    current, start = terminal.prepared.following, terminal.prepared.following.frame
    snapshot = records[0]['old_prefix_snapshot'] | dict(pending=[])
    packet = dict(closed=True, error=None, retired=None, current=parts.binding.S.encode(current))
    decisions = {start + 2: dict(reason=None, raw=[list(row) for row in parts.mode.B.grid(observed)],
                                source_call_token='synthetic-stable'),
                 start + 4: dict(reason='inactive_or_effect_window')}
    old = N(current=current, steps={start + i: dict(events=[]) for i in (2, 4)}, decisions=decisions,
        pending=pending, engine=parts.mode.C.T, S=parts.binding.S, packet=packet,
        origin_adapter=N(snapshot=lambda: snapshot))
    replay = V.Replay(parts, terminal.services, old, {}, (), records)
    replay.handoff, replay.ledger = start, first_ledger
    replay.lane = terminal.services.lane.Lane(parts, current, terminal.prepared.ledger_after,
        '予告陽性と観測後の終端30着弾に条件付け。較正済み保証ではない。')
    replay.used_terminal.update((0, 1))
    replay.terminal_frame(start + 2)
    assert old.pending == []
    replay.ledger = binding.ledger
    replay.terminal_frame(start + 4)
    result = replay.finish(dict(kind='second_prefix_finish', **snapshot), start + 4)
    assert result['acknowledged'] == (11 if late_ack else 10) and result['applied'] == 11
    assert len(result['unacknowledged_terminal_tokens']) == (0 if late_ack else 1)
    assert len(binding.mode.applied) == 1
