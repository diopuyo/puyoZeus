"""実Mode接続候補。旧live_adapterと衝突しない専用module名を使う。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import identified_origin_candidate as O
import lane_state as N
import stable_evidence as E
import stable_decision as D
import prefix_origin_initial as I

SOURCE = Path(__file__).resolve().parent.parent / 'g2_arrival_ack_candidate_2026-09-12_v1/arrival_mode.py'
ASSUMPTION = 'ちぎり補完後world内の未較正一様prior。手数間の混合priorは定義しない。'


class Live:
    def __init__(self, stack: Any, module: Any) -> None:
        self.stack, self.module = stack, module
        self.parts = SimpleNamespace(mode=module,
            arrival_saved=SimpleNamespace(V=SimpleNamespace(R=module.Q.R)))
        self.recorder = E.Recorder(stack, module)
        self.lanes: dict[int, tuple[Any, N.Lane]] = {}
        self.active: dict[int, tuple] = {}
        self.streams: dict[Path, Any] = {}
        stack.callback(self.release)

    def release(self) -> None:
        self.lanes.clear()
        self.active.clear()
        self.streams.clear()

    def write(self, mode: Any, name: str, row: dict) -> None:
        path = mode.state['output'] / name
        if path not in self.streams:
            self.streams[path] = self.stack.enter_context(path.open('x', encoding='utf-8'))
        self.streams[path].write(json.dumps(row, allow_nan=False) + '\n')
        self.streams[path].flush()

    def lane(self, mode: Any) -> N.Lane | None:
        entry = self.lanes.get(id(mode))
        self.module.B.require(entry is None or entry[0] is mode, 'prefix_mode_identity')
        return None if entry is None else entry[1]

    def observe(self, original: Any, mode: Any, item: dict, result: Any, error: Any) -> dict:
        require = self.module.B.require
        require(id(mode) not in self.active, 'prefix_reentrant_call')
        self.active[id(mode)] = (mode, item, result)
        try:
            return original(mode, item, result, error)
        finally:
            self.active.pop(id(mode), None)

    def source(self, mode: Any, item: dict) -> dict:
        owned = self.active.get(id(mode))
        self.module.B.require(owned is not None and owned[0] is mode and owned[1] is item and owned[2] is not None,
            'prefix_missing_active_call')
        self.recorder.owner(mode, item)
        return item['scope'] | dict(kind='step', token=item['token'], events=deepcopy(item['events']),
            software_reset=item['epoch'], status='returned', exception=None)

    def stable(self, original: Any, mode: Any, item: dict, result: Any) -> tuple:
        observed, reason = original(mode, item, result)
        self.source(mode, item)
        decision = D.capture(self.module, mode, item, result, reason)
        self.write(mode, 'PREFIX_STABLE_DECISIONS.jsonl', decision)
        if reason is None:
            self.recorder.record(mode, item, mode.stable_qualification)
        return observed, reason

    def capture(self, original: Any, mode: Any, item: dict) -> None:
        lane = self.lane(mode)
        if lane is None:
            return original(mode, item)
        step = self.source(mode, item)
        for event in step['events']:
            origin = event.get('active_origin')
            if origin is None:
                continue
            identity = (origin['object_id'], origin['trigger_sec'])
            if identity in mode.origin_ids:
                saved = mode.origins[mode.origin_ids[identity]]
                self.module.B.require(origin['before_board'] is not None
                    and tuple(map(tuple, origin['before_board']['grid'])) == saved['grid'],
                    'arrival_origin_mutated')
            else:
                self.origin(mode, lane, step, origin, identity)

    def origin(self, mode: Any, lane: N.Lane, step: dict, origin: dict, identity: tuple) -> None:
        ledger, require = mode.arrival_ledger, self.module.B.require
        grid, trigger = O.source_origin(self.parts, ledger, step, origin)
        eligible = [a for a in ledger.arrivals[len(ledger.applied):] if a.frame / O.FPS <= trigger]
        candidate = None
        if len(eligible) == 1:
            arrival = eligible[0]
        else:
            candidate = O.select(self.parts, ledger, lane.base, lane.families, step, origin)
            arrival = ledger.arrivals[candidate['absolute_arrival_index']]
        old = mode.origins.get(arrival.token)
        require(old is None or old['grid'] == grid, 'prefix_origin_alias_grid_changed')
        record = dict(arrival_token=arrival.token, source_call_token=step['token'], origin=origin,
            candidate=candidate, kind='first' if old is None else 'additional',
            assignment_branch='single_eligible' if candidate is None else 'identified_prepop',
            physical_origin_certified=False, quality_gate_clear=False)
        self.write(mode, 'PREFIX_ORIGIN_ASSIGNMENTS.jsonl', record)
        if old is None:
            mode.origins[arrival.token] = dict(grid=grid, source_call_token=step['token'],
                first_observed_frame=step['frame_idx'], object_id=origin['object_id'],
                estimated_chain_count=origin['chain_count'], creation_call_witnessed=False)
        mode.origin_ids[identity] = arrival.token
        lane.firing(ledger, arrival.token)

    def progress(self, original: Any, mode: Any, item: dict, result: Any, row: dict) -> dict:
        lane = self.lane(mode)
        if lane is None:
            result_row = original(mode, item, result, row)
            if mode.basis_cascade_closed and mode.arrival_ledger.applied:
                current = mode.connection.registry.current(mode.connection.binding)
                lane = N.Lane(self.parts, current, mode.arrival_ledger, ASSUMPTION)
                self.lanes[id(mode)] = (mode, lane)
                self.write(mode, 'PREFIX_LANE_INITIAL.jsonl', I.encode(self.module.BASE.V1.S, mode, lane))
            return result_row
        self.source(mode, item)
        self.module.B.require(mode.connection.registry.current(mode.connection.binding) is lane.current,
            'prefix_live_current_changed')
        observed, reason = mode.stable(item, result)
        if reason is not None:
            return row | dict(reason='arrival:' + reason)
        evidence = self.condition(mode, lane, item, observed)
        self.write(mode, 'PREFIX_LANE_OBSERVATIONS.jsonl', evidence)
        prepared = lane.prepare(mode.arrival_ledger, item['token'])
        if prepared is None:
            return row | dict(reason='prefix_ambiguous_prepop_or_no_new_arrival', provisional_update=False)
        return self.commit(mode, lane, prepared, row)

    def condition(self, mode: Any, lane: N.Lane, item: dict, observed: Any) -> dict:
        """支持0/上限超過は保存して元例外を再送出。理由だけの成功にしない。"""
        try:
            return lane.observe(mode.arrival_ledger, item['scope']['frame_idx'], observed, item['token'])
        except BaseException as error:
            try:
                saved = dict(source_call_token=item['token'], frame=item['scope']['frame_idx'],
                    error=repr(error), observed=self.module.B.grid(observed), ledger=asdict(mode.arrival_ledger),
                    base=lane.base, fired_tokens=sorted(lane.fired),
                    current=self.module.BASE.V1.S.encode(lane.current),
                    families=[N.V.encode(self.module.BASE.V1.S, family, frozenset()) for family in lane.families],
                    inputs_are_private_hypotheses=True, continue_permission=False, quality_gate_clear=False)
                self.write(mode, 'PREFIX_LANE_FAILURES.jsonl', saved)
            except BaseException as save_error:
                print('PREFIX_FAILURE_SAVE_ERROR=' + repr(save_error), file=sys.stderr, flush=True)
            raise

    def commit(self, mode: Any, lane: N.Lane, prepared: Any, row: dict) -> dict:
        c = mode.connection
        following, receipt = c.registry.transition(c.recovery.factory, c.binding, prepared.expected,
            prepared.receipt['source_call_token'], lambda current: (prepared.following, prepared.receipt))
        mode.arrival_ledger = prepared.ledger_after
        lane.accepted(prepared, following)
        mode.applied.append(receipt)
        return row | dict(reason=receipt['kind'] + '_applied', transition=receipt, provisional_update=True,
            stable_qualification=mode.stable_qualification)

    def close(self, original: Any, mode: Any) -> None:
        """Mode終了でsidecarを閉じる。外側scope終了まで開きっぱなしにしない。"""
        body, failure = sys.exc_info()[1], None
        try:
            original(mode)
        except BaseException as error:
            failure = error
        try:
            output = mode.state['output']
            streams = {path: stream for path, stream in (self.streams | self.recorder.streams).items()
                if path.parent == output}
            for stream in streams.values():
                stream.close()
            lane = self.lane(mode)
            with (output / 'PREFIX_LANE_STATUS.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(closed=all(value.closed for value in streams.values()),
                    files=sorted(path.name for path in streams), initialized=lane is not None,
                    observations=0 if lane is None else len(lane.observations),
                    commits=0 if lane is None else len(lane.committed),
                    error=mode.error, close_error=None if failure is None else repr(failure),
                    quality_gate_clear=False), stream, allow_nan=False)
        except BaseException:
            if failure is None and body is None:
                raise
        if failure is not None and body is None:
            raise failure


def install(stack: Any, module: Any, replace: Any) -> Live:
    cls = module.Mode
    module.B.require(Path(cls.capture_origin.__code__.co_filename).resolve() == SOURCE,
        'prefix_original_mode_source')
    state = Live(stack, module)
    observe, stable, capture, progress = cls.observe, cls.stable, cls.capture_origin, cls.progress
    close = cls.close
    def on_observe(mode: Any, item: dict, result: Any, error: Any) -> dict:
        return state.observe(observe, mode, item, result, error)
    def on_stable(mode: Any, item: dict, result: Any) -> tuple:
        return state.stable(stable, mode, item, result)
    def on_capture(mode: Any, item: dict) -> None:
        return state.capture(capture, mode, item)
    def on_progress(mode: Any, item: dict, result: Any, row: dict) -> dict:
        return state.progress(progress, mode, item, result, row)
    def on_close(mode: Any) -> None:
        state.close(close, mode)
    for name, method in (('observe', on_observe), ('stable', on_stable),
        ('capture_origin', on_capture), ('progress', on_progress), ('close', on_close)):
        replace(stack, cls, name, method)
    return state
