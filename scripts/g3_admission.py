"""当該frameの既存OCR観測を採録資格へ結ぶ。raw盤面や物理状態は変更しない。"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable

from scripts.g3_agent_review import VERIFY, encoded, save

SIDES = ('1P', '2P')
KEY = 'g3_ui_admission'
UNKNOWN, OBSERVED = 'UI_UNKNOWN', 'UI_OBSERVED'


def require(condition: bool, reason: str) -> None:
    """観測来歴の不整合を黙って資格ありへ変換しない。"""
    if not condition:
        raise ValueError('g3_admission:' + reason)


class Admission:
    """一つのpipelineと当該frameだけを所有。過去の陽性値は繰り越さない。"""

    def __init__(self, output: Path, source_id: str) -> None:
        require(output.resolve().is_relative_to(VERIFY.resolve()), 'output_root')
        require(type(source_id) is str and bool(source_id), 'source_required')
        self.output = output
        self.source_id, self.run_id = source_id, output.resolve().as_posix()
        self.stream = (output / 'G3_UI_ADMISSION.jsonl').open('xb')
        self.pipe: Any = None
        self.active: dict[str, Any] | None = None
        self.latest: dict[str, Any] | None = None
        self.frames = self.decisions = self.withheld = 0
        self.error: str | None = None
        self.save_error: str | None = None
        self.closed = False
        self.references: list[tuple[Any, str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        """既存のcanonical JSON表現でD原票を逐次保存する。"""
        self.stream.write(encoded(row | dict(source_id=self.source_id, run_id=self.run_id)) + b'\n')
        self.stream.flush()

    def begin(self, pipe: Any, frame: int, clock: float, image: Any) -> None:
        require(not self.closed and self.active is None and self.error is None, 'update_reentry_or_failed')
        require(type(frame) is int and frame >= 0, 'frame_type')
        require(self.pipe is None or self.pipe is pipe, 'pipeline_owner')
        require(self.latest is None or self.latest['frame'] < frame, 'duplicate_or_old_frame')
        self.pipe = pipe
        self.latest = None
        self.active = dict(frame=frame, time_sec=clock, image=image,
                           scores={side: None for side in SIDES}, reads={side: 0 for side in SIDES})

    def observe(self, tracker: Any, image: Any, result: tuple[Any, ...]) -> None:
        active = self.active
        require(active is not None and active['image'] is image, 'score_outside_owned_frame')
        if tracker is None:
            return
        sides = [side for side in SIDES if tracker is getattr(self.pipe, '_score_tracker_' + side.lower())]
        require(len(sides) == 1, 'tracker_owner')
        side = sides[0]
        require(active['reads'][side] == 0, 'duplicate_score_read')
        require(type(result) is tuple and len(result) == 4, 'score_helper_schema')
        raw = result[1]
        require(raw is None or type(raw) is int and raw >= 0, 'score_raw_type')
        active['reads'][side] += 1
        active['scores'][side] = raw

    def finish(self, error: BaseException | None) -> None:
        active, self.active = self.active, None
        require(active is not None, 'finish_without_update')
        row = {key: value for key, value in active.items() if key != 'image'}
        row.update(kind='ui_observation', status=UNKNOWN, quality_gate_clear=False)
        if error is not None:
            self.error = row['error'] = repr(error)
        elif any(score is not None for score in row['scores'].values()):
            row['status'] = OBSERVED
        self.latest = row
        self.frames += 1
        try:
            self.write(row)
        except BaseException as failure:
            self.save_error = repr(failure)
            if error is None:
                self.error = repr(failure)
                raise

    def decide(self, frame: int, clock: float, side: str, original: bool) -> bool:
        row = self.latest
        require(self.error is None and self.active is None and row is not None, 'admission_before_update')
        require(row['frame'] == frame and row['time_sec'] == clock and side in SIDES, 'admission_frame')
        allowed = bool(original and row['status'] == OBSERVED)
        self.decisions += 1
        self.withheld += int(original and not allowed)
        self.write(dict(kind='admission', frame=frame, time_sec=clock, side=side,
                        original_eligible=bool(original), allowed=allowed, ui_status=row['status'],
                        quality_gate_clear=False))
        return allowed

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        if self.closed:
            return False
        restored = all(vars(owner).get(name) is old for owner, name, old in self.references)
        failure: BaseException | None = None
        if body is not None:
            self.error = self.error or repr(body)
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as error:
            self.save_error = repr(error)
            failure = error
        try:
            self.stream.close()
        except BaseException as error:
            self.save_error = repr(error)
            failure = failure or error
        self.closed = self.stream.closed
        try:
            save(self.output / 'G3_UI_ADMISSION_STATUS.json', dict(closed=self.closed, frames=self.frames,
                 decisions=self.decisions, withheld=self.withheld, error=self.error,
                 save_error=self.save_error, references_restored=restored, source_id=self.source_id,
                 run_id=self.run_id, quality_gate_clear=False))
        except BaseException as error:
            self.save_error = repr(error)
            failure = failure or error
        if failure is not None and body is None:
            raise failure
        require(restored or body is not None, 'references_not_restored')
        return False


def replace_static(stack: Any, cls: Any, name: str, value: staticmethod) -> None:
    """getattrで関数へ変換せず、所有する静的descriptorだけを復元する。"""
    previous = vars(cls)[name]
    setattr(cls, name, value)
    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if vars(cls).get(name) is value:
            setattr(cls, name, previous)
        elif body is None:
            raise RuntimeError('g3_admission:foreign_static_descriptor:' + name)
        return False
    stack.push(restore)


def install(stack: Any, collector: Any, state: dict[str, Any], replace: Callable[..., Any],
            *, source_id: str) -> Admission:
    """既存replace_ownedを使い、metadataとは別の3参照だけを所有する。"""
    require(KEY not in state, 'duplicate_install')
    cls = collector.RecognitionPipeline
    emit, update = collector._should_emit, cls.update
    descriptor = vars(cls)['_update_score_tracker']
    require(isinstance(descriptor, staticmethod), 'score_helper_staticmethod')
    require(emit.__globals__ is vars(collector), 'collector_globals_owner')
    observer = Admission(state['output'], source_id)
    state[KEY] = observer
    observer.references = [(cls, 'update', vars(cls)['update']),
                           (cls, '_update_score_tracker', descriptor), (collector, '_should_emit', emit)]
    stack.push(observer.close)
    def wrapped_update(pipe: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        try:
            observer.begin(pipe, frame_idx, time_sec, frame)
        except BaseException as error:
            observer.error = repr(error)
            raise
        try:
            result = update(pipe, frame_idx, time_sec, frame)
        except BaseException as error:
            observer.finish(error)
            raise
        observer.finish(None)
        return result
    def wrapped_score(tracker: Any, frame: Any, want_detail: bool = False) -> Any:
        result = descriptor.__func__(tracker, frame, want_detail=want_detail)
        observer.observe(tracker, frame, result)
        return result
    def wrapped_emit(*args: Any, **kwargs: Any) -> bool:
        # 物理持続フィルタ等の副作用を含め、元判定を必ず一回実行する。
        value = emit(*args, **kwargs)
        caller = sys._getframe(1)
        if caller.f_globals is not vars(collector) or caller.f_code.co_name != '_process_side_lean':
            return value
        values = caller.f_locals
        return observer.decide(values['frame_idx'], values['t_sec'], values['side_label'], value)
    replace(stack, cls, 'update', wrapped_update)
    replace_static(stack, cls, '_update_score_tracker', staticmethod(wrapped_score))
    replace(stack, collector, '_should_emit', wrapped_emit)
    return observer
