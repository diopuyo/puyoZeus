"""新postcommit返却の直後へ独立した既存collector比較を接続する。"""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
KEY = 'postcommit_publication_consumer'
SIDES = (('1P','p1'),('2P','p2'))
COMPARISON = 'POSTCOMMIT_CONSUMER_COMPARISON.json'
ROWS = 'POSTCOMMIT_CONSUMER_ROWS.json'
STATUS = 'POSTCOMMIT_CONSUMER_STATUS.json'
CONNECTED = ROOT.parent/'g2_publication_consumer_runtime_2026-09-09_v1/connected.py'
CONNECTED_SHA = '65f664798b8f910b2deaa59d4c67145b694c06dbeabedb78c8ef208233f26cbe'
DEATH = ROOT.parent/'g2_death_availability_boundary_2026-09-09_v1/adapter.py'
DEATH_SHA = 'cb15008d8212f97956bd01b2252485ec31ed8a609c4c427d56f359c87ffaf161'
BASE = ROOT.parent/'g2_current_consumer_release_2026-09-09_v1/consumer_comparison.py'
BASE_SHA = '232a69c933761200358f2bf429fb01a7a465bc07910b5f0620b0d9260e5e7e61'
BOUNDARY = 'postcommit_finish_before_outer_return'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)



def snapshot(value: Any) -> Any:
    """型を保つ値コピー。未知型をrepr等へ落とさず拒否する。"""
    import numpy as np
    kind = type(value)
    name = kind.__module__ + '.' + kind.__qualname__
    if isinstance(value, Enum):
        return [name, snapshot(value.value)]
    if value is None or kind in (bool, int, str):
        return [name, value]
    if kind is float:
        return [name, value.hex()]
    if isinstance(value, np.generic):
        return [name, snapshot(value.item())]
    if kind is np.ndarray:
        require(not value.dtype.hasobject, 'postcommit_object_array')
        return [name, value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if kind in (list, tuple):
        return [name, [snapshot(v) for v in value]]
    if kind is dict:
        pairs = [[snapshot(k), snapshot(v)] for k,v in value.items()]
        return [name, sorted(pairs, key=lambda p: json.dumps(p[0],sort_keys=True))]
    if is_dataclass(value) and not isinstance(value,type):
        values = {f.name:getattr(value,f.name) for f in fields(value)}
        values.update(vars(value) if hasattr(value,'__dict__') else {})
        return [name, snapshot(values)]
    module = sys.modules.get(kind.__module__)
    allowed = (('src.board','Board'),('src.probabilistic_board','ProbabilisticBoard'))
    if (kind.__module__,kind.__name__) in allowed and getattr(module,kind.__name__,None) is kind:
        return [name, snapshot(vars(value))]
    raise TypeError('postcommit_snapshot_unsupported:' + name)


def side_snapshots(result: Any) -> Any:
    return {side:snapshot(getattr(result,attr)) for side,attr in SIDES}


def foreign_snapshot(result: Any) -> Any:
    return snapshot({f.name:getattr(result,f.name) for f in fields(result) if f.name not in ('p1','p2')})

def comparison(receiver: Any) -> Any:
    old = receiver.consumer.publisher.comparison
    require(tuple(side for side,_ in SIDES)==tuple(receiver.r.B.SIDES), 'postcommit_side_order')
    require(sha(BASE)==BASE_SHA and Path(type(old).__mro__[1].__init__.__code__.co_filename).resolve()==BASE,
            'postcommit_base_comparison')
    require(all(not lane.frames for lane in old.lanes.values()), 'postcommit_comparison_late')
    cls, lane = type(old), old.lanes['held']
    require(Path(cls.__init__.__code__.co_filename).resolve() == CONNECTED and sha(CONNECTED)==CONNECTED_SHA,
            'postcommit_comparison_class')
    death = sys.modules[type(lane.death).observe_result.__module__]
    require(Path(death.__file__).resolve()==DEATH and sha(DEATH)==DEATH_SHA, 'postcommit_death_module')
    observer = lane.death._module
    pipeline = sys.modules[lane.death._pipeline_type.__module__]
    require(pipeline.PipelineResult is lane.death._pipeline_type
            and pipeline.SideResult is lane.death._side_type, 'postcommit_pipeline_types')
    value = cls(lane.C, receiver.registration, death, observer, pipeline)
    require(all(value.lanes[k] is not old.lanes[k] and value.lanes[k].acc is not old.lanes[k].acc
                and all(getattr(value.lanes[k],n) is not getattr(old.lanes[k],n)
                    for n in ('death','states','shared','ojama','physical','previous'))
                for k in old.lanes), 'postcommit_distinct_lanes')
    return value


class Consumer:
    def __init__(self, receiver: Any, compared: Any, output: Path) -> None:
        self.receiver, self.comparison, self.output = receiver, compared, output
        self.rows: list[Any] = []
        self.errors: list[str] = []
        self.active, self.closed = False, False
        self.registration = snapshot(receiver.registration)

    def serialized(self, result: Any) -> Any:
        return {side:self.receiver.rec.side_value(getattr(result,attr)) for side,attr in SIDES}

    def joined(self, result: Any, pipe: Any) -> Any:
        old = self.receiver.consumer.publisher.entries
        require(len(old)==len(self.rows)+1, 'postcommit_old_entry_count')
        entry = old[-1]
        require(result.frame_idx in self.receiver.rec.expected and
            (entry['frame_idx'],entry['time_sec'])==(result.frame_idx,result.time_sec), 'postcommit_old_clock')
        require(entry['returned']==self.serialized(result), 'postcommit_old_return_join')
        tsumo = tuple(pipe.tsumo_count(side) for side,_ in SIDES)
        require(list(tsumo)==entry['actual_tsumo_getter'], 'postcommit_old_tsumo_join')
        return pipe._post_match_lockdown_active,tsumo

    def finish(self, original: Callable[[Any], Any], result: Any) -> Any:
        require(not self.active and not self.closed and not self.errors, 'postcommit_consumer_state')
        self.active = True
        try:
            context = self.receiver.active
            require(context is not None, 'postcommit_update_missing')
            pipe = context['pipe']
            lock,tsumo = self.joined(result,pipe)
            before, full_before = self.serialized(result),side_snapshots(result)
            foreign_before = foreign_snapshot(result)
            released_before = self.receiver.released
            final = original(result)
            published, full_after = self.serialized(final),side_snapshots(final)
            require(side_snapshots(result)==full_before and foreign_snapshot(result)==foreign_before,
                    'postcommit_original_mutated')
            foreign = {f.name:getattr(final,f.name) for f in fields(final) if f.name not in ('p1','p2')}
            require(foreign_snapshot(final)==foreign_before, 'postcommit_foreign_changed')
            require((lock,tsumo)==self.joined(result,pipe), 'postcommit_pipe_changed')
            self.comparison.consume(result,final,lock,tsumo)
            require(full_before==side_snapshots(result) and full_after==side_snapshots(final)
                and foreign_snapshot(result)==foreign_before and foreign_snapshot(final)==foreign_before
                and all(getattr(final,k) is v for k,v in foreign.items()), 'postcommit_consumer_mutation')
            require(self.registration==snapshot(self.receiver.registration), 'postcommit_registration_changed')
            self.rows.append(dict(frame_idx=final.frame_idx,time_sec=final.time_sec,
                before=before,after=published,full_before=full_before,full_after=full_after,
                actual_tsumo_getter=list(tsumo),post_match_lockdown_active=lock,
                changed_sides=[s for s,_ in SIDES if full_before[s]!=full_after[s]],
                projection_changed_sides=[s for s,_ in SIDES if before[s]!=published[s]],
                issued_total=self.receiver.issued, tickets_this_update=len(context['tickets']),
                released_this_update=self.receiver.released-released_before,
                same_result_identity=final is result, old_projection_join_verified=True,
                comparison_completed=True, actual_collector_append_verified=False,
                observation_boundary=BOUNDARY,quality_gate_clear=False))
            return final
        except BaseException as exc:
            self.errors.append(repr(exc))
            raise
        finally:
            self.active = False

    def close(self) -> None:
        require(not self.active and not self.closed, 'postcommit_consumer_close')
        if self.rows and not self.errors:
            value = self.comparison.snapshot()
            value.update(observation_boundary=BOUNDARY, no_event_only=not any(r['released_this_update'] for r in self.rows),
                changed_input_observed=any(r['changed_sides'] for r in self.rows), quality_gate_clear=False)
            write(self.output/COMPARISON,value)
        write(self.output/ROWS,self.rows)
        self.closed = True
        write(self.output/STATUS,dict(closed=True,errors=self.errors,updates=len(self.rows),
            quality_gate_clear=False,training_permission=False,accounting_permission=False,
            actual_collector_append_verified=False, observation_boundary=BOUNDARY))


def install(stack: Any, state: Any) -> Consumer:
    require(KEY not in state, 'postcommit_consumer_duplicate')
    receiver = state['postcommit_current_receiver']
    require(not receiver.closed and not receiver.errors and receiver.active is None
            and receiver.issued==receiver.released==0, 'postcommit_receiver_late')
    require('finish' not in vars(receiver), 'postcommit_finish_wrapped')
    consumer = Consumer(receiver,comparison(receiver),receiver.rec.output)
    original = receiver.finish
    stack.callback(consumer.close)
    stack.callback(vars(receiver).pop,'finish',None)
    receiver.finish = lambda result: consumer.finish(original,result)
    state[KEY] = consumer
    return consumer
