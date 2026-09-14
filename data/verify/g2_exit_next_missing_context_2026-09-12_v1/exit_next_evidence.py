"""原pipelineの同call値だけを資格へ写す。推論再実行・NEXT値の補完はしない。"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import CodeType
from typing import Any
import numpy as np
import next_vote_policy as P

FRAME_SHAPE = (1080,1920,3)
MAX_STACK_DEPTH = 32


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError('exit_next_evidence:'+reason)


def pixels(value: Any) -> str:
    require(type(value) is np.ndarray and value.shape == FRAME_SHAPE and value.dtype == np.uint8,'image_shape')
    return hashlib.sha256(value.tobytes()).hexdigest()


def pipeline_frame(caller: Any, code: CodeType, pipe: Any) -> Any:
    found = []
    for _ in range(MAX_STACK_DEPTH):
        if caller is None:
            break
        if caller.f_code is code:
            require(caller.f_locals.get('self') is pipe,'pipeline_identity')
            found.append(caller)
        caller = caller.f_back
    require(len(found) == 1,'original_pipeline_frame_count')
    return found[0]


@dataclass
class ImageReceipt:
    frame: int
    scope: tuple
    image: Any
    digest: str


class Provider:
    def __init__(self, qualification: Any, code: CodeType, slide_class: type, output: Path) -> None:
        self.qualification,self.code,self.slide_class = qualification,code,slide_class
        self.last: ImageReceipt | None = None
        self.rows = 0
        self.max_rows = qualification.deadline-qualification.reset_frame+1
        self.error: str | None = None
        self.path = output/'EXIT_NEXT_EVIDENCE.jsonl'
        self.stream = self.path.open('x',encoding='utf-8')

    def context(self, caller: Any, tick: Any) -> tuple[Any,dict,tuple]:
        r,q = self.qualification.recovery,self.qualification
        machine,signals = caller.f_locals['self'],caller.f_locals['signals']
        require(machine.allowed is q and machine.machine is r.pipe._sm_1p,'qualification_owner')
        item = r.journal.active
        require(item is not None and item['pipe'] is r.pipe,'active_J')
        row = q.rows[-1] if q.rows else None
        require(row is not None and row['eligible'] is True and row['same_call'] is True
                and row['frame'] == tick.frame and row['token'] == item['token'],'original_qualification')
        saved = r.waits.get(id(item['frame']))
        require(saved is not None and saved['item'] is item and saved['caller'] is item['frame'],'same_wait')
        source = pipeline_frame(caller,self.code,r.pipe).f_locals
        require(type(source.get('frame_idx')) is int and source['frame_idx'] == tick.frame
                and source.get('time_sec') == signals.time_sec == tick.clock == tick.frame/60,'clock')
        require(source['frame'] is item['frame'].f_locals.get('frame_bgr'),'current_frame_identity')
        require(source.get('next_pair_1p') == signals.next_pair == tick.next_pair,'filtered_next')
        require(source.get('slide_check_1p',False) is source.get('slide_1p') is signals.slide_motion,'slide_identity')
        scope = tuple(row['scope'])
        require(tuple(saved['scope']) == scope == tuple(r.evidence.scope(r.factory,r.pipe)),'scope')
        return r,source,scope

    def observe(self, caller: Any, tick: Any) -> tuple[Any,dict]:
        r,source,scope = self.context(caller,tick)
        current = source['frame']
        digest = pixels(current)
        prior = self.last
        self.last = ImageReceipt(tick.frame,scope,current,digest)
        row = dict(frame=tick.frame,scope=scope,filtered_next=tick.next_pair,
                   source_call_token=r.journal.active['token'],vote_decision_not_made_here=True,reason='known_next',
                   quality_gate_clear=False,physical_certified=False)
        if tick.next_pair is not None:
            return None,row
        if source.get('slide_check_1p') is not True:
            return None,row|dict(reason='no_actual_slide_suppression')
        if source.get('p1_pair') is None or source.get('both') is None:
            return None,row|dict(reason='raw_next_missing')
        if prior is None or prior.frame != tick.frame-P.OLD.OLD.FRAME_STRIDE or prior.scope != scope:
            return None,row|dict(reason='missing_consecutive_image')
        require(prior.image is r.pipe._prev_frame and pixels(prior.image) == prior.digest,'previous_frame_identity')
        slide = r.pipe._slide_detector_1p
        require(type(slide) is self.slide_class and slide.side == '1P','original_slide_detector')
        require(slide._diff_threshold == P.FIXED_THRESHOLD,'fixed_threshold_changed')
        raw = tuple(map(int,source['p1_pair']))
        require(tuple(map(int,source['both'].p1.next_pair)) == raw,'original_raw_next')
        scores = [float(np.abs(slide._extract_roi_gray(prior.image,roi)
                   -slide._extract_roi_gray(current,roi)).mean()) for roi in slide._get_rois()]
        require(len(scores) == 4 and max(scores) == slide._last_diff_score,'original_max_diff')
        proof = P.ExitEvidence(tick.frame,prior.frame,raw,source['slide_check_1p'],
                               max(scores[:2]),max(scores[2:]),float(slide._diff_threshold))
        row.update(reason='observed_exit',raw_next=raw,slide=proof.slide_motion,roi_scores=scores,
                   current_pixel_sha=digest,previous_pixel_sha=prior.digest)
        return proof,row

    def __call__(self, caller: Any, tick: Any) -> Any:
        try:
            require(self.error is None and not self.stream.closed and self.rows < self.max_rows,'provider_lifetime')
            proof,row = self.observe(caller,tick)
            # 受理は後段の原3過去票全条件で決まる。ここは観測票のみ。
            row['evidence_supplied'] = proof is not None
            self.stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            self.stream.flush()
            self.rows += 1
            return proof
        except BaseException as error:
            self.error = repr(error)
            raise

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        self.last = None
        failure = None
        try:
            self.stream.close()
        except BaseException as error:
            failure = error
        try:
            with self.path.with_name('EXIT_NEXT_EVIDENCE_STATUS.json').open('x') as stream:
                json.dump(dict(rows=self.rows,error=self.error,closed=self.stream.closed,
                               retained_image=False,quality_gate_clear=False,
                               close_error=None if failure is None else repr(failure)),stream,indent=2)
        except BaseException as error:
            failure = failure or error
        if body is None and failure is not None:
            raise failure
        return False
