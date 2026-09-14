"""原Recorder→原writer→Witness tap→終了候補。開始FIFOとstep通知は人工。"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import second_inactive_candidate as C
import test_second_inactive_candidate as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_second_terminal_arrival_2026-09-14_v1'))
import test_original_witness_tap as W


def prepare(path: Path, stack: ExitStack) -> tuple:
    end, item, _ = F.setup(path, stack)
    module, journal, pipe, _ = W.setup(path)
    stack.callback(journal.close)
    pipe._tsumo_count_2p = Counter()
    before = end.ledger
    scope = tuple(id(pipe) if i == 3 else v for i, v in enumerate(before.scope))
    ledger = replace(before, scope=scope, arrivals=tuple(replace(a, scope=scope) for a in before.arrivals))
    end.ledger = end.value.ledger = ledger
    connection = end.value.mode.connection
    connection.binding.scope = scope
    current = N(scope=scope)
    connection.registry.current = lambda _: current
    connection.recovery.journal, connection.recovery.pipe = journal, pipe
    connection.recovery.evidence.scope = lambda *args: scope
    journal.source_id, journal.run_id = scope[:2]
    journal.epoch = lambda *args: scope[2]
    journal.scope = lambda p, s, f, t: dict(source_id=scope[0], run_id=scope[1],
        pipe_object_id=id(p), side=s, frame_idx=f, time_sec=t, generation=dict(reset_epoch=scope[5]))
    journal.selected = {(f,'2P') for f in range(ledger.clock+2, ledger.deadline+2, 2)}
    journal.codes = {prepare.__code__}
    item['pipe'], item['frame'].f_code = pipe, prepare.__code__
    unacked = ledger.arrivals[len(ledger.acknowledgements):]
    pipe._pending_tsumo_2p.extend(a.pair for a in unacked)
    journal.fifo.sync(pipe, '2P', scope[2])['tokens'] = [a.token for a in unacked]
    witness = W.writer().install(stack, journal)
    end.value.tap = W.T.Tap(witness, path/'second.jsonl', stack)
    stack.push(end.close)
    return end, item, module, journal, pipe, witness


def test_original_writer_inactive_to_deadline(tmp_path: Path) -> None:
    calls = []
    def original(p: Any, s: str, f: int, t: float, active: bool, pair: Any) -> None:
        calls.append((f, active))
    with ExitStack() as stack:
        end, item, module, journal, pipe, witness = prepare(tmp_path, stack)
        original_ledger = end.ledger
        enqueue = journal.wrap_enqueue(original)
        for frame in range(end.ledger.clock+2, end.ledger.deadline+2, 2):
            queue = pipe._pending_tsumo_2p
            journal.update_before['2P'] = dict(queue=queue, refs=tuple(queue), accounting=module.account(pipe,'2P'))
            queue.clear()  # 原pipelineのinactive clearを模擬。
            enqueue(pipe, '2P', frame, frame/60, False, None)
            item['scope']['frame_idx'] = item['frame'].f_locals['frame_idx'] = frame
            item['frame'].f_locals['time_sec'] = frame/60
            item['token'] = f'synthetic-step:{frame}'
            end.observe(item, None)
        assert end.value.ledger is original_ledger and len(end.ledger.acknowledgements) == 10
        assert not journal.errors and witness.error is None and end.rows == 8
    assert len(calls) == 8 and all(active is False for _, active in calls)
    assert json.loads((tmp_path/'second_STATUS.json').read_bytes())['restored']
    assert json.loads((tmp_path/'SECOND_INACTIVE_END_STATUS.json').read_bytes())['last_frame'] == 36900
