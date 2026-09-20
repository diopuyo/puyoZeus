"""2P原経路を保持し、複数消費の時だけ私有prefixへ移る候補。"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path
from typing import Any
import consumed_prefix as C
import stable_decision as D

FPS = 60
ACTIVATION_PENDING = 2
RECEIPT_KIND = 'second_consumed_prefix/v1'


class Adapter:
    def __init__(self, mode: Any, engine: Any, serializer: Any) -> None:
        self.mode, self.engine, self.serializer = mode, engine, serializer
        self.lane: C.Lane | None = None
        self.fired: frozenset[str] = frozenset()
        self.stream: Any = None
        self.failure: str | None = None
        self.finished = False

    def write(self, value: dict) -> None:
        self.engine.B.require(not self.finished, 'second_prefix_write_after_finish')
        if self.stream is None:
            initial = self.mode.connection.binding.initial_call_token.replace(':', '_')
            path = Path(self.mode.state['output']) / ('SECOND_PREFIX_' + initial + '.jsonl')
            self.stream = path.open('x', encoding='utf-8')
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()

    def snapshot(self) -> dict:
        return dict(lane=None if self.lane is None else self.lane.snapshot(self.serializer),
            pending=[asdict(event) for event in self.mode.native.pending], fired_tokens=sorted(self.fired),
            error=self.failure, quality_gate_clear=False, original_fifo_changed=False)

    def sync(self) -> None:
        lane, mode, check = self.lane, self.mode, self.engine.B.require
        check(lane is not None, 'second_prefix_not_active')
        check(mode.connection.registry.current(mode.connection.binding) is lane.current, 'second_prefix_registry_owner')
        remaining = lane.arrivals[lane.committed:]
        check(len(mode.native.pending) >= len(remaining), 'second_prefix_pending_lost')
        for arrival, event in zip(remaining, mode.native.pending):
            check((arrival.token, arrival.pair, arrival.frame, arrival.call_token) ==
                  (event.occurrence_token, event.pair, event.consumed_frame, event.source_call_token),
                  'second_prefix_pending_identity')
        for event in mode.native.pending[len(remaining):]:
            lane.append(mode.connection.binding.scope, event)

    def activate(self, item: dict) -> None:
        mode, check = self.mode, self.engine.B.require
        if self.lane is not None or len(mode.native.pending) < ACTIVATION_PENDING:
            return
        check(mode.basis_origin is None or mode.basis_cascade_closed, 'second_prefix_basis_not_settled')
        current = mode.connection.registry.current(mode.connection.binding)
        self.lane = C.Lane(self.engine, current)
        self.sync()
        for arrival in self.lane.arrivals:
            if arrival.token in mode.origins:
                self.positive_origin(mode.origins[arrival.token]['grid'])
                self.fired |= frozenset((arrival.token,))
        self.write(dict(kind='second_prefix_initial', source_call_token=item['token'],
                        frame=item['scope']['frame_idx'], **self.snapshot()))

    def positive_origin(self, grid: tuple) -> None:
        simulator = self.engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
        result = simulator.simulate(self.engine.B.Board.from_dict({'grid': grid}))
        self.engine.B.require(result.chain_count > 0, 'second_prefix_nonpositive_origin')

    def origin(self, item: dict, origin: dict) -> None:
        mode, lane, check = self.mode, self.lane, self.engine.B.require
        trigger = origin['trigger_sec']
        check(type(trigger) in (int,float) and math.isfinite(trigger)
              and lane.initial.frame/FPS <= trigger <= item['scope']['time_sec'], 'second_prefix_origin_clock')
        check(origin['before_board'] is not None, 'second_prefix_origin_missing')
        check(type(origin['object_id']) is int and origin['object_id']>0, 'second_prefix_origin_identity')
        grid = self.engine.B.grid(self.engine.B.Board.from_dict({'grid':origin['before_board']['grid']}))
        identity = (origin['object_id'], trigger)
        if identity in mode.origin_ids:
            check(mode.origins[mode.origin_ids[identity]]['grid'] == grid, 'second_prefix_origin_mutated')
            return
        self.positive_origin(grid)
        eligible = [a for a in lane.arrivals[lane.committed:] if a.frame/FPS <= trigger]
        if len(eligible) == 1:
            arrival = eligible[0]
        else:
            check(bool(lane.families) and all(f.phase == C.P.PREPOP and f.prefix > lane.committed
                and f.value.frame/FPS <= trigger and all(w.grid[self.engine.B.HIDDEN_ROWS:] ==
                grid[self.engine.B.HIDDEN_ROWS:] for w in f.value.worlds) for f in lane.families),
                'second_prefix_origin_ambiguous')
            prefixes = {f.prefix for f in lane.families}
            check(len(prefixes) == 1, 'second_prefix_origin_prefix_ambiguous')
            arrival = lane.arrivals[next(iter(prefixes))-1]
        old = mode.origins.get(arrival.token)
        check(old is None or old['grid'] == grid, 'second_prefix_origin_alias_mutated')
        if old is None:
            mode.origins[arrival.token] = dict(grid=grid, source_call_token=item['token'],
                first_observed_frame=item['scope']['frame_idx'], object_id=origin['object_id'],
                estimated_chain_count=origin['chain_count'], creation_call_witnessed=False)
        mode.origin_ids[identity] = arrival.token
        self.fired |= frozenset((arrival.token,))
        self.write(dict(kind='second_prefix_origin', source_call_token=item['token'],
            frame=item['scope']['frame_idx'], occurrence_token=arrival.token, origin=origin,
            physical_certified=False, quality_gate_clear=False))

    def progress(self, item: dict, result: Any, row: dict) -> dict:
        self.sync()
        observed, reason = self.mode.stable(item, result)
        if reason is not None:
            return row | dict(reason='second_prefix:'+reason, provisional_update=False)
        self.lane.condition(item['scope']['frame_idx'], observed, self.fired)
        self.write(dict(kind='second_prefix_observation', source_call_token=item['token'],
            frame=item['scope']['frame_idx'], observed=self.engine.B.grid(observed), **self.snapshot()))
        prepared = self.lane.prepare(item['token'])
        if prepared is None:
            return row | dict(reason='second_prefix_ambiguous_prepop_or_no_new_hand', provisional_update=False)
        return self.commit(prepared, row)

    def commit(self, prepared: C.Prepared, row: dict) -> dict:
        mode, lane, check = self.mode, self.lane, self.engine.B.require
        tokens = tuple(a.token for a in prepared.arrivals)
        check(tokens == tuple(e.occurrence_token for e in mode.native.pending[:len(tokens)]),
              'second_prefix_commit_pending')
        receipt = dict(kind=RECEIPT_KIND, source_call_token=prepared.call_token,
            applied_frame=prepared.following.frame, applied_tokens=tokens,
            consumed_arrivals=[asdict(a) for a in prepared.arrivals],
            before_prefix=prepared.before_prefix, after_prefix=prepared.after_prefix,
            state=self.serializer.encode(prepared.following), physical_certified=False,
            original_fifo_changed=False, quality_gate_clear=False)
        c = mode.connection
        following, _ = c.registry.transition(c.recovery.factory,c.binding,prepared.expected,
            prepared.call_token,lambda current:(prepared.following,receipt))
        lane.accept(prepared,following)
        del mode.native.pending[:len(tokens)]
        mode.applied.append(receipt)
        self.write(dict(kind='second_prefix_commit', receipt=receipt, **self.snapshot()))
        return row | dict(reason=RECEIPT_KIND, transition=receipt, provisional_update=True,
            pending_occurrences=[e.occurrence_token for e in mode.native.pending])

    def finish(self) -> None:
        if self.finished:
            return
        try:
            if self.stream is not None:
                self.write(dict(kind='second_prefix_finish', **self.snapshot()))
        finally:
            if self.stream is not None:
                self.stream.close()
            self.finished = True


def mode_class(base: type, engine: Any, serializer: Any) -> type:
    class Mode(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.prefix = Adapter(self,engine,serializer)

        def observe(self, item: Any, result: Any, error: Any) -> dict:
            try:
                return super().observe(item,result,error)
            except BaseException as failure:
                self.prefix.failure = repr(failure)
                try:
                    self.prefix.write(dict(kind='second_prefix_failure', source_call_token=item['token'],
                        frame=item['scope']['frame_idx'], **self.prefix.snapshot()))
                except BaseException as save_error:
                    print('SECOND_PREFIX_SAVE_ERROR='+repr(save_error),file=sys.stderr)
                raise

        def capture_origin(self, item: dict) -> None:
            self.prefix.activate(item)
            if self.prefix.lane is None:
                return super().capture_origin(item)
            self.prefix.sync()
            for event in item['events']:
                if event.get('active_origin') is not None:
                    self.prefix.origin(item,event['active_origin'])

        def stable(self, item: Any, result: Any) -> tuple:
            observed, reason = super().stable(item,result)
            self.prefix.write(D.capture(engine,self,item,result,reason))
            return observed, reason

        def follow_hand(self, item: Any, result: Any, row: dict) -> dict:
            if self.prefix.lane is None:
                return super().follow_hand(item,result,row)
            return self.prefix.progress(item,result,row)
    return Mode
