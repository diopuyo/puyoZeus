"""2P sidecarを原J/原通常手数学/元STABLE資格と突合する。"""
from __future__ import annotations
from dataclasses import asdict
from types import SimpleNamespace as N
from typing import Any
import json
import consumed_prefix as C
import physical_adapter as A
import stable_decision as D

STRIDE = 2


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value,allow_nan=False))


def row_frame(row: dict) -> int | None:
    if row['kind'] == 'second_prefix_finish':
        return None
    return row.get('frame',row.get('scope',{}).get('frame_idx',row.get('receipt',{}).get('applied_frame')))


class Replay:
    def __init__(self, engine: Any, serializer: Any, native: Any, original: Any,
                 packet: dict, steps: dict, contexts: dict) -> None:
        self.engine,self.S,self.native,self.original = engine,serializer,native,original
        self.packet,self.steps,self.contexts = packet,steps,contexts
        self.current = serializer.decode(packet['initial']['state'])
        self.pending: list[Any] = []
        self.seen: set[str] = set()
        self.lane: C.Lane | None = None
        self.origin_adapter: Any = None
        self.decisions: dict[int,dict] = {}
        self.receipts = {r['applied_frame']:r for r in packet['applied']}
        engine.B.require(len(self.receipts)==len(packet['applied']), 'second_replay_duplicate_receipt')
        self.commits: set[int] = set()

    def begin(self, row: dict, step: dict) -> None:
        check = self.engine.B.require
        check(self.lane is None and row['source_call_token']==step['token'], 'second_replay_initial_call')
        check(row['lane']['initial']==self.S.encode(self.current)
              and row['pending']==normalized([asdict(e) for e in self.pending]), 'second_replay_initial_state')
        self.lane = C.Lane(self.engine,self.current)
        self.sync()
        check(not row['fired_tokens'], 'second_replay_unwitnessed_initial_fire')
        mode = N(origins={},origin_ids={},connection=N(binding=N(scope=self.current.scope)))
        adapter = A.Adapter(mode,self.engine,self.S)
        adapter.lane = self.lane
        adapter.write = lambda value: None
        self.origin_adapter = adapter
        check(row['lane']==normalized(self.lane.snapshot(self.S)), 'second_replay_initial_snapshot')

    def sync(self) -> None:
        lane, check = self.lane,self.engine.B.require
        remaining = lane.arrivals[lane.committed:]
        check(len(self.pending)>=len(remaining), 'second_replay_pending_lost')
        check(tuple(a.token for a in remaining)==tuple(e.occurrence_token for e in self.pending[:len(remaining)]),
              'second_replay_pending_prefix')
        for event in self.pending[len(remaining):]:
            lane.append(lane.current.scope,event)

    def origin(self, row: dict, step: dict) -> None:
        check = self.engine.B.require
        check(self.lane is not None and row['source_call_token']==step['token'], 'second_replay_origin_call')
        check(row['origin'] in [e.get('active_origin') for e in step['events']], 'second_replay_origin_J')
        self.sync()
        emitted = []
        self.origin_adapter.write = emitted.append
        scope = dict(time_sec=step['time_sec'],frame_idx=step['frame_idx'])
        self.origin_adapter.origin(dict(token=step['token'],scope=scope),row['origin'])
        check(normalized(emitted)==[row], 'second_replay_origin_assignment')

    def observation(self, row: dict, step: dict) -> None:
        check = self.engine.B.require
        frame = step['frame_idx']
        decision = self.decisions.get(frame)
        check(self.lane is not None and row['source_call_token']==step['token']
              and decision is not None and decision['reason'] is None, 'second_replay_observation_qualified')
        check(row['observed']==decision['raw'], 'second_replay_observation_board')
        self.sync()
        observed = self.engine.B.Board.from_dict({'grid':row['observed']})
        self.lane.condition(frame,observed,self.origin_adapter.fired)
        check(row['lane']==normalized(self.lane.snapshot(self.S))
              and row['pending']==normalized([asdict(e) for e in self.pending]), 'second_replay_observation_snapshot')

    def commit(self, row: dict, step: dict) -> None:
        check = self.engine.B.require
        prepared = self.lane.prepare(step['token'])
        check(prepared is not None and step['frame_idx'] not in self.commits, 'second_replay_commit_once')
        receipt = row['receipt']
        check(receipt==self.receipts.get(step['frame_idx']) and receipt['kind']==A.RECEIPT_KIND
              and receipt['source_call_token']==step['token'], 'second_replay_commit_packet')
        tokens = tuple(a.token for a in prepared.arrivals)
        check(receipt['applied_tokens']==list(tokens) and receipt['consumed_arrivals']==normalized(
            [asdict(a) for a in prepared.arrivals]), 'second_replay_commit_arrivals')
        check(receipt['before_prefix']==prepared.before_prefix and receipt['after_prefix']==prepared.after_prefix
              and receipt['state']==self.S.encode(prepared.following), 'second_replay_commit_math')
        check(all(receipt[key] is False for key in ('physical_certified','original_fifo_changed','quality_gate_clear')),
              'second_replay_commit_authority')
        check(tuple(e.occurrence_token for e in self.pending[:len(tokens)])==tokens, 'second_replay_commit_head')
        self.lane.accept(prepared,prepared.following)
        self.current = prepared.following
        del self.pending[:len(tokens)]
        self.commits.add(step['frame_idx'])
        check(row['lane']==normalized(self.lane.snapshot(self.S)) and row['pending']==normalized(
            [asdict(e) for e in self.pending]), 'second_replay_commit_snapshot')

    def ordinary(self, receipt: dict, step: dict) -> None:
        check = self.engine.B.require
        check(self.lane is None and receipt.get('origin') is None, 'second_replay_ordinary_scope')
        check(self.decisions.get(step['frame_idx'],{}).get('reason','missing') is None,
              'second_replay_ordinary_qualification')
        encoded = normalized([asdict(e) for e in self.pending])
        self.original.apply(encoded,receipt,step)
        event = self.pending[0]
        observed = self.engine.B.Board.from_dict({'grid':step['returned']['confirmed']['grid']})
        prior = self.engine.H.uncalibrated_uniform('原列挙配置をworld内で未較正一様と仮定。着手頻度ではない。')
        following,report = self.engine.run(self.current,self.current.scope,step['frame_idx'],
            event.occurrence_token,event.pair,observed,0,prior,origin_observed=None)
        check(self.S.encode(following)==receipt['state'] and normalized(asdict(report))==receipt['distribution_report'],
              'second_replay_ordinary_math')
        self.pending.pop(0)
        self.current = following

    def record(self, row: dict, step: dict) -> None:
        kind, check = row['kind'],self.engine.B.require
        if kind == D.VERSION:
            frame = step['frame_idx']
            check(frame not in self.decisions, 'second_replay_duplicate_decision')
            D.verify(self.engine,row,step,self.contexts[frame])
            self.decisions[frame] = row
        elif kind == 'second_prefix_initial':
            self.begin(row,step)
        elif kind == 'second_prefix_origin':
            self.origin(row,step)
        elif kind == 'second_prefix_observation':
            self.observation(row,step)
        elif kind == 'second_prefix_commit':
            self.commit(row,step)
        else:
            check(False,'second_replay_unknown_or_failure_record')

    def run(self, records: list[dict], end: int) -> dict:
        check = self.engine.B.require
        check(records and records[-1]['kind']=='second_prefix_finish', 'second_replay_finish_missing')
        check(self.packet['closed'] and self.packet['error'] is None,'second_replay_packet_lifetime')
        first = self.original.step_for(self.steps,self.current.frame,list(self.current.scope))
        check(first['token']==self.packet['initial']['source_call_token'],'second_replay_initial_J')
        groups: dict[int,list] = {}
        previous = self.current.frame
        for row in records[:-1]:
            frame = row_frame(row)
            check(type(frame) is int and self.current.frame<frame and previous<=frame<=end
                  and (frame-self.current.frame)%STRIDE==0,'second_replay_record_order')
            groups.setdefault(frame,[]).append(row)
            previous = frame
        initial = self.current.frame
        scope = list(self.current.scope)
        for frame in range(initial+STRIDE,end+STRIDE,STRIDE):
            step = self.original.step_for(self.steps,frame,scope)
            event = self.native.extract(dict(token=step['token'],scope=dict(frame_idx=frame),events=step['events']))
            if event is not None:
                check(event.occurrence_token not in self.seen,'second_replay_duplicate_consumption')
                self.seen.add(event.occurrence_token)
                self.pending.append(event)
            for row in groups.get(frame,()):
                self.record(row,step)
            receipt = self.receipts.get(frame)
            if receipt is not None and receipt.get('kind')!=A.RECEIPT_KIND:
                self.ordinary(receipt,step)
            elif receipt is not None:
                check(frame in self.commits,'second_replay_missing_commit')
        finish = records[-1]
        check(finish['error'] is None and finish['pending']==normalized([asdict(e) for e in self.pending]),
              'second_replay_finish_pending')
        check(finish['lane']==(None if self.lane is None else normalized(self.lane.snapshot(self.S))), 'second_replay_finish_lane')
        retired = self.packet.get('retired')
        if retired is None:
            check(self.packet['current']==self.S.encode(self.current), 'second_replay_final_current')
        else:
            check(retired['frame']==end+STRIDE and self.packet['current'] is None
                  and retired['old_state']==self.S.encode(self.current), 'second_replay_retired_current')
            self.original.retirement(retired,self.steps[retired['frame']],
                normalized([asdict(e) for e in self.pending]),scope)
        return dict(consumed=len(self.seen),prefix_commits=len(self.commits),pending=len(self.pending),
            normal_receipts=len(self.receipts)-len(self.commits),quality_gate_clear=False)
