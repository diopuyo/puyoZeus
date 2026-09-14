"""原採録完了境界でSessionを駆動し、固定pipeline.updateには触れない。"""
from __future__ import annotations
import json
from typing import Any,Callable

KEYS=('probabilistic_basis_connection','probabilistic_tracking_mode')


class Boundary:
    def __init__(self,stack: Any,context: Any,create: Callable[...,Any]) -> None:
        self.stack,self.context,self.create=stack,context,create
        self.loop,self.pipe,self.state=(context[k] for k in ('collector','pipe','state'))
        self.original=self.loop.collect_lean
        self.update=type(self.pipe).update
        assert self.loop.raw_bound.class_update is self.update,'collector_session_after_raw_install'
        self.session: Any=None
        self.frame: int | None=None
        self.restored=False
        self.error: Any=None
        self.wrapper=self.collect
        stack.push(self.finish)
        self.loop.collect_lean=self.wrapper

    def collect(self,cap: Any,pipeline: Any,start_frame: int,n_frames: int,
                effective_interval_frames: int,fps: float) -> Any:
        result=self.original(cap,pipeline,start_frame,n_frames,effective_interval_frames,fps)
        if pipeline is not self.pipe: return result
        try:
            assert n_frames==1 and fps==60,'collector_session_single_update'
            assert type(self.pipe).update is self.update,'collector_session_pipeline_changed'
            if self.session is None:
                if all(k in self.state for k in KEYS):
                    self.session=self.create(self.stack,self.context)
                    self.state['belief_m1_session']=self.session
                    self.frame=start_frame
                    self.stack.push(self.close_restore)
                    assert start_frame not in self.session.frames,'collector_session_first_evaluation'
            else: self.session.completed(start_frame)
        except BaseException as error:
            self.error=error
            raise
        return result

    def restore(self) -> None:
        assert type(self.pipe).update is self.update,'collector_session_pipeline_changed'
        if not self.restored:
            assert self.loop.collect_lean is self.wrapper,'collector_session_foreign_hook'
            self.loop.collect_lean=self.original
            self.restored=True
        assert self.loop.collect_lean is self.original,'collector_session_restore_changed'
        if self.session is not None: self.session.restored=True

    def close_restore(self,kind: Any,body: Any,trace: Any) -> bool:
        try: self.restore()
        except BaseException as error:
            self.error=error
            if body is None: raise
        return False

    def finish(self,kind: Any,body: Any,trace: Any) -> bool:
        try:
            self.restore()
            if body is None:
                assert self.session is not None and self.error is None,'collector_session_incomplete'
        except BaseException as error:
            self.error=error
            if body is None: raise
        finally:
            packet=dict(installed=self.session is not None,frame=self.frame,restored=self.restored,
                pipeline_update_unchanged=type(self.pipe).update is self.update,
                error=None if self.error is None else repr(self.error),
                body_error=None if body is None else repr(body),quality_gate_clear=False)
            try:
                with (self.state['output']/'BELIEF_M1_COLLECTOR_ATTACH.json').open('x',encoding='utf-8') as stream:
                    json.dump(packet,stream)
            except BaseException:
                if body is None: raise
        return False
