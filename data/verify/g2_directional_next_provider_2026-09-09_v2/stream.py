"""既存幾何エンジンの初期静穏と連続scopeを管理する。会計ownerではない。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

INITIAL_QUIET_COMPARISONS = 2


class Stream:
    def __init__(self, candidate: Any, types: Any, source: str, side: str) -> None:
        self.c, self.types, self.source, self.side = candidate, types, source, side
        self.segment = 0
        self.epoch: int | None = None
        self.last_frame: int | None = None
        self.engine: Any = None
        self.quiet_run = 0
        self.latest: Any = None
        self.last_record: dict[str, Any] | None = None

    def invalidate(self) -> None:
        self.segment += 1
        self.epoch, self.last_frame, self.engine = None, None, None
        self.quiet_run, self.latest, self.last_record = 0, None, None

    def sequence(self) -> str:
        return f'{self.side}:epoch:{self.epoch}:segment:{self.segment}'

    def start(self, invocation: Any, epoch: int) -> Any:
        self.c.require(type(epoch) is int and epoch >= 0, 'epoch')
        self.epoch, self.last_frame = epoch, invocation.frame
        self.latest = self.observation(invocation, None, None)
        return self.latest

    def observation(self, invocation: Any, quiet: Any, event: Any) -> Any:
        return self.types.MotionObservation(invocation=invocation, side=self.side,
            software_epoch=self.epoch, segment_id=self.sequence(), frame_idx=invocation.frame,
            time_sec=invocation.time_sec, quiet_frames=quiet, candidate=event)

    def update(self, invocation: Any, packet: dict[str, Any]) -> Any:
        c = self.c
        c.require(packet['frame_after'] == invocation.frame
                  and packet['time_after'] == invocation.time_sec, 'invocation_clock')
        scope = c.Scope(self.source, self.sequence())
        before = {'last_frame': self.last_frame, 'last_time': self.last_frame / c.FPS}
        c.validate_packet(packet, scope, self.side, before)
        stats = c.slot_statistics(packet['slots'])
        quiet = all(stats[name]['quiet_support'] for name in c.SLOTS)
        run = self.quiet_run + 1 if quiet else 0
        event, detail = None, None
        if self.engine is not None:
            detail = self.engine.update(packet)
            value = detail['candidate']
            if value is not None:
                event = self.types.MotionCandidate(sequence_number=value['sequence_number'],
                    first_support_frame=min(r['frame_after'] for r in value['supporting_slots'].values()),
                    available_frame=value['available_frame'], reference=detail['packet_digest'])
        elif run >= INITIAL_QUIET_COMPARISONS:
            initial = c.InitialArming(invocation.frame, invocation.time_sec, True,
                'same_scope_actual_points_all_slots_two_quiet_comparisons', False)
            self.engine = c.Detector(scope, self.side, initial)
        frames = (invocation.frame - c.STRIDE, invocation.frame) if run >= INITIAL_QUIET_COMPARISONS else None
        self.latest = self.observation(invocation, frames, event)
        self.last_record = {'statistics': deepcopy(stats), 'detector': deepcopy(detail)}
        self.last_frame, self.quiet_run = invocation.frame, run
        return self.latest
