"""原enqueue保存＋人工step通知の終端consumer対照。実動画/元finalizer完走ではない。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import first_terminal_saved as V
import test_first_terminal_candidate as F

END, BASIS = 14, 10


def fixture(path: Path) -> tuple:
    mode, journal, pipe, capture = F.setup(path)
    mode.arrival_ledger = replace(mode.arrival_ledger, deadline=END)
    registry = mode.connection.registry
    registry.value = replace(registry.value, deadline=END)
    with ExitStack() as stack:
        mode.arrival_capture = capture(mode, path)
        stack.push(mode.arrival_capture.close)
        for frame in (12, 14):
            F.send(journal, pipe, frame)
            F.notify(mode, journal, pipe, frame)
    enqueues = [json.loads(line) for line in journal.stream.getvalue().splitlines()]
    packets = [json.loads(line) for line in (path/'FIRST_TERMINAL.jsonl').read_text().splitlines()]
    raw = [dict(kind='step', side='1P', frame_idx=BASIS, code_sha256='synthetic-step-code')]
    for index, (enqueue, packet) in enumerate(zip(enqueues, packets, strict=True)):
        enqueue['row_index'] = index*2
        step = dict(enqueue, kind='step', token=packet['completed_call_token'], row_index=index*2+1,
            code_sha256='synthetic-step-code', events=[], exception=None,
            generation_after=deepcopy(packet['generation_after']))
        raw.extend((enqueue, step))
    original = N(L=F.F.A.L, R=N(N=F.F.A.CORE.Mode.observe.__globals__['N']))
    contexts = {BASIS: mode.state['provisional_context_observer'].rows[0]}
    return original, mode.arrival_ledger, raw, packets, contexts


def test_tail_coverage_and_no_additional_consumption(tmp_path: Path) -> None:
    report = V.verify_records(*fixture(tmp_path))
    assert report['terminal_end'] == END and report['original_terminal_updates'] == 2
    assert report['additional_acknowledgements'] == report['additional_physical_applications'] == 0
    assert not report['quality_gate_clear']


@pytest.mark.parametrize('native_clock', [18, 22])
def test_native_ledger_clock_mismatch_before_read(native_clock: int) -> None:
    mode = N(arrival_capture=object(), native=N(last_frame=native_clock), arrival_ledger=N(clock=20))
    with pytest.raises(ValueError, match='terminal_saved_live_clock'):
        V.terminal_files(N(), {'probabilistic_tracking_mode':mode})


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'active', 'code', 'source', 'generation', 'receipt', 'extra'])
def test_reject_incomplete_or_changed_tail(tmp_path: Path, mutation: str) -> None:
    original, ledger, raw, packets, contexts = fixture(tmp_path)
    if mutation == 'missing': raw.pop()
    elif mutation == 'failed': raw[-1]['exception'] = 'original failure'
    elif mutation == 'active': raw[-2]['active'] = True
    elif mutation == 'code': raw[-1]['code_sha256'] = 'foreign'
    elif mutation == 'source': packets[-1]['source_fields']['token'] = 'foreign'
    elif mutation == 'generation': packets[-1]['generation_after']['reset_epoch'] += 1
    elif mutation == 'receipt': packets[-1]['terminal_receipt'] = packets[0]['terminal_receipt']
    elif mutation == 'extra': raw.append(dict(raw[-1], token='extra'))
    with pytest.raises(ValueError):
        V.verify_records(original, ledger, raw, packets, contexts)
