"""既存同call journalと実NEXT Invocationへのread-only参照。追加認識しない。"""
from __future__ import annotations

import copy
import functools
import hashlib
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import transaction as T

ROOT = Path(__file__).resolve().parent
JOURNAL = ROOT.parent / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
JOURNAL_SHA = 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'
NEXT = ROOT.parents[2] / 'scripts/next_enqueue_live_shadow_v1.py'
NEXT_SHA = 'e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237'
SOURCE = 'sha256:b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
FIRST, LAST, STRIDE, FPS, SIDE = 34796, 34910, 2, 60, '1P'


def source_matches(value: Any, path: Path, expected: str) -> None:
    init = type(value).__init__
    actual = Path(init.__code__.co_filename).resolve()
    T.require(actual == path.resolve() and Path(init.__globals__['__file__']).resolve() == actual
        and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
        'provider_source:' + path.name)
    module = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
    cls = next(code for code in module.co_consts
        if isinstance(code, CodeType) and code.co_name == type(value).__name__)
    expected_code = next(code for code in cls.co_consts if isinstance(code, CodeType) and code.co_name == '__init__')
    T.require(init.__code__ == expected_code and init.__code__.co_linetable == expected_code.co_linetable
        and repr(init.__code__.co_consts) == repr(expected_code.co_consts), 'provider_constructor_code')


class Provider:
    def __init__(self, journal: Any) -> None:
        source_matches(journal, JOURNAL, JOURNAL_SHA)
        source_matches(journal.controller, NEXT, NEXT_SHA)
        T.require(journal.source_id == SOURCE, 'provider_video_scope')
        self.journal = journal
        self.enqueues: dict[str, Any] = {}
        self.invocations: dict[int, Any] = {}
        self.controller: Any = None

    def selected(self, pipe: Any, side: str, frame: int, clock: float) -> bool:
        return side == SIDE and type(frame) is int and FIRST <= frame <= LAST

    def view(self, pipe: Any, side: str, frame: int, clock: float, caller: Any) -> T.View:
        rec, item = self.journal, self.journal.active
        T.require(clock == frame / FPS and frame % STRIDE == 0, 'completion_clock')
        T.require(item is not None and item['frame'] is caller and caller.f_code in rec.codes
            and item['pipe'] is pipe and not rec.errors, 'completion_actual_step')
        scope = rec.scope(pipe, side, frame, clock)
        T.require(item['scope'] == scope and item['epoch'] == rec.epoch(pipe, side), 'journal_scope_changed')
        values = caller.f_locals
        T.require(values.get('self') is pipe and values.get('side') == side
            and values.get('frame_idx') == frame and values.get('time_sec') == clock
            and values.get('sm') is getattr(pipe, '_sm_' + side.lower()), 'caller_locals')
        invocation = rec.controller._invocation(pipe, frame, clock)
        actual_next = type(rec.controller).__init__.__globals__['_actual_next_result']
        T.require(invocation.error is None and actual_next(invocation.main), 'next_return_missing')
        observed = getattr(invocation.main, 'p1' if side == '1P' else 'p2')
        pair, dnext = observed.next_pair, observed.dnext_pair
        quiet, _ = rec.controller._quiet(invocation, side, pair)
        T.require(id(caller) not in self.invocations, 'duplicate_provider_view')
        self.invocations[id(caller)] = (invocation, invocation.main, invocation.slides[side])
        old = self.owner(pipe, side, item['epoch'])
        enqueue = self.enqueues.get(side)
        self.check_enqueue(enqueue, scope, item['epoch'], old)
        identity = (scope['source_id'], scope['run_id'], item['epoch'], id(pipe),
            id(getattr(pipe, '_sm_' + side.lower())), scope['generation']['reset_epoch'], side)
        return T.View(identity, frame, clock, old['queue'], old['refs'], tuple(old['tokens']),
            pair, dnext, quiet, tuple(enqueue['added_occurrence_tokens']))

    def owner(self, pipe: Any, side: str, epoch: int) -> dict[str, Any]:
        old = self.journal.fifo.entries.get((id(pipe), side))
        queue = getattr(pipe, '_pending_tsumo_' + side.lower())
        T.require(old is not None and old['queue'] is queue and old['epoch'] == epoch, 'actual_fifo_owner')
        T.require(len(old['refs']) == len(queue) == len(old['tokens'])
            and all(a is b for a, b in zip(old['refs'], queue)), 'actual_fifo_refs')
        return old

    def check_enqueue(self, row: Any, scope: Any, epoch: int, owner: Any) -> None:
        T.require(type(row) is dict and row['status'] == 'returned'
            and row['active'] is True and row['software_reset'] == epoch, 'fresh_enqueue_missing')
        T.require(all(row[k] == v for k, v in scope.items()), 'enqueue_same_update_scope')
        added, tokens = row['added_occurrence_tokens'], row['fifo_occurrence_tokens']
        T.require(tokens == owner['tokens'] and not row['discarded_tokens'], 'enqueue_owner_tokens')
        T.require(len(added) <= 1 and all(type(t) is str and t for t in added), 'enqueue_added_shape')
        before, after = row['before']['pending_tsumo'], row['after']['pending_tsumo']
        if added:
            expected = row['token'] + ':slot:' + str(len(before))
            T.require(added == [expected] and tokens[-1] == expected and tokens.count(expected) == 1,
                'enqueue_actual_slot')
            T.require(len(after) == len(before) + 1 and after[:-1] == before
                and after[-1] == row['after']['last_consumed_color'], 'enqueue_added_values')
        else:
            T.require(before == after, 'no_append_mutated_fifo')

    def before_consume(self, ticket: Any, caller: Any) -> None:
        view, item = ticket['view'], ticket['item']
        values = caller.f_locals
        rec, pipe, side = self.journal, values['self'], values['side']
        T.require(rec.active is not None and rec.active['frame'] is caller
            and rec.active['scope']['frame_idx'] == view.frame, 'prepared_actual_caller')
        now = self.owner(pipe, side, view.scope[2])
        T.require(now['queue'] is view.queue and tuple(now['tokens']) == view.tokens
            and len(now['refs']) == len(view.refs) and all(a is b for a, b in zip(now['refs'], view.refs)),
            'prepared_owner_changed')
        invocation = rec.controller._invocation(pipe, view.frame, view.clock)
        bound, main, slide = self.invocations[id(caller)]
        T.require(invocation is bound and invocation.main is main and invocation.slides[side] is slide
            and invocation.error is None, 'prepared_next_identity_changed')
        T.require(T.board_key(values['prev_confirmed']) == T.board_key(item.baseline)
            and T.board_key(values['ctx'].confirmed_board) == item.previous, 'prepared_boards_changed')

    def after_consume(self, ticket: Any, caller: Any) -> None:
        rec, item = self.journal, ticket['item']
        T.require(rec.active is not None and rec.active['frame'] is caller, 'consume_actual_caller')
        events = [r for r in rec.active['events'] if r['stage'] == 'fifo_after']
        T.require(len(events) == 1 and events[0]['enqueue_occurrence_token'] == item.token,
            'actual_journal_consume_token')

    def after_step(self, ticket: Any, caller: Any) -> None:
        values, item = caller.f_locals, ticket['item']
        owner = self.owner(values['self'], values['side'], ticket['view'].scope[2])
        T.require(item.token not in owner['tokens'] and tuple(owner['tokens']) == ticket['view'].tokens[1:],
            'completed_slot_not_removed_exactly_once')
        pending = values.get('landing_pending')
        if pending is not None and pending[0] == values['frame_idx']:
            T.require(values.get('falling_pair_for_grace') is item.pair, 'grace_not_committed_pair')
        T.require(not self.journal.errors, 'journal_failed')

    def attach(self, stack: Any, controller: Any) -> None:
        rec, self.controller = self.journal, controller
        emit, complete = rec.emit, rec.complete_step

        def captured(row: Any) -> None:
            emit(row)
            if row.get('kind') == 'enqueue':
                self.enqueues[row['side']] = copy.deepcopy(row)

        @functools.wraps(complete)
        def finished(item: Any, result: Any, error: Any, profile: Any) -> None:
            try:
                if item['frame'] is not None:
                    controller.finish(item['frame'], error)
            except BaseException as failure:
                rec.errors.append('completion:' + repr(failure))
                if error is None:
                    raise
            finally:
                if item['frame'] is not None:
                    self.invocations.pop(id(item['frame']), None)
                complete(item, result, error, profile)
        rec.emit, rec.complete_step = captured, finished
        stack.callback(setattr, rec, 'emit', emit)
        stack.callback(setattr, rec, 'complete_step', complete)
