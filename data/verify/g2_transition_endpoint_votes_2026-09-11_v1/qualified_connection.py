"""原selectedで保存済みの同call資格だけをSM入口で読む。完了後の値は借りない。"""
from __future__ import annotations
import json
import sys
from typing import Any
import prior_votes_v2 as V

SIDE = '1P'
FPS = 60


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('qualified_prior_votes:' + reason)


class Qualification:
    def __init__(self, recovery: Any, reset_frame: int, deadline: int) -> None:
        require(type(reset_frame) is int and type(deadline) is int and reset_frame < deadline,
                'clock_order')
        self.recovery, self.reset_frame, self.deadline = recovery, reset_frame, deadline
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None

    def __call__(self) -> bool:
        r = self.recovery
        if r.pending is None:
            return False
        caller = sys._getframe(1)
        try:
            require(caller.f_code is V.PriorVotes.update.__code__, 'vote_update_caller')
            transport = caller.f_back
            require(transport.f_code is r.control._parts.C.Controller.update.__code__, 'transport_code')
            require(transport.f_locals.get('self') is r.control, 'transport_identity')
            frame, signals = caller.f_locals['frame'], caller.f_locals['signals']
            require(transport.f_locals.get('sm') is r.pipe._sm_1p
                    and transport.f_locals.get('signals') is signals, 'transport_arguments')
            return self.check(frame, signals, transport.f_locals['caller'])
        except BaseException as error:
            self.error = repr(error)
            raise

    def check(self, frame: int, signals: Any, caller: Any) -> bool:
        r, item = self.recovery, self.recovery.journal.active
        require(r.error is None and not r.journal.errors and r.reset_count == 1, 'prior_failure')
        require(item is not None and item['frame'] is caller and item['pipe'] is r.pipe
                and caller.f_code in r.journal.codes, 'actual_J')
        require(type(frame) is int and signals.time_sec == frame / FPS, 'clock')
        if not self.reset_frame < frame <= self.deadline:
            return False
        scope = r.evidence.scope(r.factory, r.pipe)
        require(scope[2] == r.pending['epoch'] and not r.pending['used'], 'pending_epoch')
        require(item['scope'] == r.journal.scope(r.pipe, SIDE, frame, signals.time_sec)
                and item['epoch'] == r.journal.epoch(r.pipe, SIDE) == scope[2], 'J_scope')
        require(scope[:2] == r.pending['old_scope'][:2]
                and scope[3:5] == r.pending['old_scope'][3:5]
                and scope[5] > r.pending['old_scope'][5] and scope[6] == SIDE, 'generation')
        saved = r.waits.get(id(caller))
        require(saved is not None and saved['item'] is item and saved['caller'] is caller
                and saved['scope'] == scope and saved['frame'] == frame
                and saved['clock'] == signals.time_sec, 'same_call_wait')
        require(not r.pipe._pending_tsumo_1p and id(caller) not in r.control.tickets, 'FIFO_ticket')
        view = saved['view']
        eligible = view is not None and r.provider.no_origin(r.pipe, SIDE, view)
        self.rows.append(dict(frame=frame, token=item['token'], scope=scope, eligible=eligible,
                              view_present=view is not None, same_call=True))
        return eligible


def install(stack: Any, recovery: Any, state: dict[str, Any], reset_frame: int, deadline: int) -> Any:
    require('qualified_prior_votes' not in state, 'duplicate_install')
    qualification = Qualification(recovery, reset_frame, deadline)
    value = V.install(stack, recovery.pipe._sm_1p, qualification)
    state['qualified_prior_votes'] = value
    def close() -> None:
        with (state['output'] / 'QUALIFIED_PRIOR_VOTES.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(qualifications=qualification.rows, votes=value.rows,
                error=qualification.error, quality_gate_clear=False), stream, indent=2)
    stack.callback(close)
    return value
