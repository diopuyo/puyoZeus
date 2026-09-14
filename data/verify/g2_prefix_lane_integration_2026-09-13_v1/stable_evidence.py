"""成功/保留に依存せずQ2資格票を別保存する。元tracking行は変更しない。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

VERSION = 'prefix-stable-qualification/v1'


class Recorder:
    def __init__(self, stack: Any, module: Any) -> None:
        self.stack, self.module = stack, module
        self.owners: dict[int, tuple] = {}
        self.last: dict[int, tuple[int, str]] = {}
        self.streams: dict[Path, Any] = {}
        stack.callback(self.release)

    def release(self) -> None:
        self.owners.clear()
        self.last.clear()
        self.streams.clear()

    def owner(self, mode: Any, item: dict) -> tuple:
        c, require = mode.connection, self.module.B.require
        ledger, scope = mode.arrival_ledger, item['scope']
        self.module.L.check(ledger)
        sm = item['frame'].f_locals['sm']
        journal = c.recovery.journal
        require(item['frame'].f_code in journal.codes and journal.active is None
            and item['frame'].f_locals['frame_idx'] == scope['frame_idx'], 'prefix_stable_original_frame')
        actual = (mode, c, c.binding, c.recovery.factory, c.recovery.pipe, sm)
        prior = self.owners.get(id(mode))
        require(prior is None or all(old is new for old, new in zip(prior, actual)),
            'prefix_stable_owner_changed')
        require(c.binding.scope == ledger.scope and item['pipe'] is c.recovery.pipe
            and id(item['pipe']) == ledger.scope[3] and id(sm) == ledger.scope[4],
            'prefix_stable_live_owner')
        expected = ledger.scope
        require(scope['source_id'] == expected[0] and scope['run_id'] == expected[1]
            and item['epoch'] == expected[2] and scope['pipe_object_id'] == expected[3]
            and scope['generation']['reset_epoch'] == expected[5]
            and scope['generation']['side'] == scope['side'] == expected[-1], 'prefix_stable_scope')
        require(ledger.clock <= scope['frame_idx'] <= ledger.deadline, 'prefix_stable_deadline')
        return actual

    def record(self, mode: Any, item: dict, qualification: dict) -> dict:
        actual = self.owner(mode, item)
        require = self.module.B.require
        self.module.Q.validate(qualification)
        require(qualification['source_call_token'] == item['token']
            and qualification['scope'] == json.loads(json.dumps(item['scope']))
            and qualification['software_reset'] == item['epoch'], 'prefix_stable_capture_call')
        previous = self.last.get(id(mode))
        frame = item['scope']['frame_idx']
        require(previous is None or (previous[0] < frame and previous[1] != item['token']),
            'prefix_stable_duplicate_or_order')
        row = dict(kind=VERSION, journal_token=item['token'], stable_qualification=deepcopy(qualification),
            owner_scope=mode.arrival_ledger.scope, owner_ids=[id(owner) for owner in actual],
            committed=False, original_tracking_changed=False, quality_gate_clear=False)
        path = mode.state['output'] / 'PREFIX_STABLE_QUALIFICATIONS.jsonl'
        if path not in self.streams:
            self.streams[path] = self.stack.enter_context(path.open('x', encoding='utf-8'))
        stream = self.streams[path]
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        self.owners[id(mode)], self.last[id(mode)] = actual, (frame, item['token'])
        return row


def verify(parts: Any, row: dict, step: dict, context: dict, ledger: Any) -> bool:
    """独立原J/contextに元Q2を適用。live所有id自体の再現は主張しない。"""
    module = parts.mode
    module.B.require(row['kind'] == VERSION and row['committed'] is False
        and row['original_tracking_changed'] is False and row['quality_gate_clear'] is False,
        'prefix_stable_saved_schema')
    module.B.require(tuple(row['owner_scope']) == ledger.scope, 'prefix_stable_saved_owner')
    parts.arrival_saved.V.R.scope(step, ledger)
    return module.Q.verify(row, step, context)
