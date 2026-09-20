"""原O.emit成功と原J追加を同call参照で結合。tokenは原Jだけが作る。"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import inspect
from typing import Any
from fixed import require

SIDES, FPS, PAIR_SIZE = ('1P', '2P'), 60, 2


class Link:
    def __init__(self, parts: Any, journal: Any, adapter: Any) -> None:
        base = parts.V.D.V
        base.source_matches(journal, base.JOURNAL, base.JOURNAL_SHA)
        base.source_matches(journal.controller, base.NEXT, base.NEXT_SHA)
        require(type(adapter) is parts.O.Adapter and adapter.enabled is True
                and adapter.controller is journal.controller, 'handoff_adapter_controller')
        require(journal.source_id.removeprefix('sha256:') == adapter.source_id
                and journal.run_id == adapter.run_id, 'handoff_source_run')
        self.parts, self.journal, self.adapter = parts, journal, adapter
        self.latest: dict[str, dict[str, Any]] = {}
        self.used: set[str] = set()
        self.error: str | None = None
        self.attached = False

    def capture(self, pipe: Any, side: str, frame: int, reason: str, committed: bool) -> None:
        require(self.error is None and side in SIDES and type(committed) is bool, 'handoff_capture_fault')
        control = self.adapter.controller
        inv = control._invocation(pipe, frame, frame / FPS)
        state = self.adapter.states[(id(pipe), side)]
        history = inv.runtime.histories[side]
        require(inv.error is None and history.clock == frame and state['epoch'] == history.epoch, 'handoff_emit_clock')
        require(committed is (reason == 'occurrence_committed'), 'handoff_emit_reason')
        queue = getattr(pipe, '_pending_tsumo_' + side.lower())
        self.latest[side] = {'invocation': inv, 'history': history, 'pipe': pipe, 'queue': queue,
            'refs': tuple(queue), 'frame': frame, 'clock': frame / FPS, 'epoch': history.epoch,
            'segment': state['segment'], 'accepted': history.accepted, 'dnext': state['baseline_dnext'],
            'baseline_frame': state['baseline_frame'], 'pending': state['pending'], 'armed_at': state['armed_at'],
            'event': state['last_candidate'], 'reason': reason, 'committed': committed}

    def attach(self, stack: ExitStack) -> None:
        require(not self.attached, 'handoff_already_attached')
        original = self.adapter.emit
        require(inspect.ismethod(original) and original.__func__ is self.parts.O.Adapter.emit,
                'handoff_original_emit')
        present, previous = 'emit' in vars(self.adapter), vars(self.adapter).get('emit')
        def emitted(pipe: Any, side: str, frame: int, reason: str, committed: bool) -> Any:
            try:
                result = original(pipe, side, frame, reason, committed)
                self.capture(pipe, side, frame, reason, committed)
                return result
            except BaseException as exc:
                self.error = repr(exc)
                raise
        def restore() -> None:
            setattr(self.adapter, 'emit', previous) if present else vars(self.adapter).pop('emit', None)
            self.attached = False
        stack.callback(restore)
        self.adapter.emit, self.attached = emitted, True

    def current(self, view: Any) -> dict[str, Any]:
        require(self.attached and self.error is None and not self.journal.errors, 'handoff_failed_recording')
        side = view.scope[-1]
        require(side in SIDES and side in self.latest, 'handoff_missing_side')
        row = self.latest[side]
        inv, pipe = row['invocation'], row['pipe']
        require(view.frame == row['frame'] and view.clock == row['clock'], 'handoff_same_call')
        expected = (self.journal.source_id, self.journal.run_id, row['epoch'], id(pipe),
                    id(getattr(pipe, '_sm_' + side.lower())))
        require(view.scope[:5] == expected, 'handoff_scope')
        actual = self.adapter.controller._invocation(pipe, view.frame, view.clock)
        require(actual is inv and inv.error is None and inv.runtime.histories[side] is row['history'], 'handoff_invocation_history')
        state = self.adapter.states[(id(pipe), side)]
        require(state['segment'] == row['segment'] and state['epoch'] == row['epoch']
                and state['blocked'] is None, 'handoff_segment_epoch')
        require(view.queue is row['queue'] and len(view.refs) == len(row['refs'])
                and all(a is b for a, b in zip(view.refs, row['refs'])), 'handoff_fifo_refs')
        return row

    def basis(self, view: Any) -> tuple[Any, Any]:
        row = self.current(view)
        require(self.parts.T.valid_pair(row['accepted']) and self.parts.T.valid_pair(row['dnext']), 'handoff_basis_unknown')
        return row['accepted'], row['dnext']

    def proof(self, item: Any, view: Any, enqueue: Any) -> dict[str, Any] | None:
        if not view.added:
            return None
        row = self.current(view)
        require(len(view.refs) == len(view.tokens) == PAIR_SIZE and len(view.added) == 1, 'handoff_two_slots')
        require(item.scope == view.scope and item.queue is view.queue and item.pair is view.refs[0]
                and item.token == view.tokens[0], 'handoff_old_slot')
        require(view.tokens[-1] == view.added[0] and view.added[0] != item.token
                and view.added[0] not in self.used, 'handoff_new_token_duplicate')
        require(row['committed'] is True and row['reason'] == 'occurrence_committed'
                and row['baseline_frame'] == view.frame and row['pending'] is None and row['armed_at'] is None,
                'handoff_successful_commit_missing')
        event = row['event']
        require(type(event) is self.parts.O.MotionCandidate and event.available_frame <= view.frame
                and event.first_support_frame >= item.started, 'handoff_candidate_clock')
        require(view.quiet and all(self.parts.T.valid_pair(v) for v in (view.next_pair, view.dnext_pair))
                and item.dnext_pair == view.next_pair and view.refs[-1] == item.next_pair
                and row['accepted'] == view.next_pair and row['dnext'] == view.dnext_pair
                and item.started < view.frame, 'handoff_successor_basis')
        rec, side = self.journal, view.scope[-1]
        scope = rec.scope(row['pipe'], side, view.frame, view.clock)
        require(view.scope[5] == scope['generation']['reset_epoch'], 'handoff_SM_reset')
        owner = rec.fifo.entries[(id(row['pipe']), side)]
        require(tuple(owner['tokens']) == view.tokens and owner['queue'] is view.queue
                and len(owner['refs']) == len(view.refs)
                and all(a is b for a, b in zip(owner['refs'], view.refs)), 'handoff_J_owner_identity')
        self.parts.V.Provider.check_enqueue(self, enqueue, scope, row['epoch'], owner)
        require(tuple(enqueue['added_occurrence_tokens']) == view.added, 'handoff_J_added')
        return {'kind': 'directional_commit_original_J/v1', 'source_id': rec.source_id, 'run_id': rec.run_id,
            'side': side, 'software_epoch': row['epoch'], 'segment_id': row['segment'],
            'available_frame': view.frame, 'available_time': view.clock, 'old_token': item.token,
            'new_token': view.added[0], 'journal_call_token': enqueue['token'],
            'candidate': deepcopy(vars(event)), 'old_accepted': item.next_pair,
            'new_accepted': row['accepted'], 'dnext': row['dnext'], 'physical_progress_certified': False}

    def consume(self, proof: dict[str, Any], view: Any) -> None:
        require(proof['available_frame'] == view.frame and proof['side'] == view.scope[-1]
                and proof['new_token'] == view.added[0] and proof['new_token'] not in self.used,
                'handoff_consumed_twice')
        self.used.add(proof['new_token'])
