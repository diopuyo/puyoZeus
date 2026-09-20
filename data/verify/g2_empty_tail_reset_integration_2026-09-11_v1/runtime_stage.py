"""保存後の全J・一回の公開集計・rolling結合を、最終実factoryへ照合する。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import sys
from typing import Any
import run_qualification as Q
import boundary_runtime as B
import stage_boundary as BOUNDARY

ADAPTER = Q.ROOT/'multiscope_stage/adapt.py'
ADAPTER_SHA = 'f133e45de3c812f3c447464fbd18f6949c3324500617b1862a861003cba55e53'
PRIVATE = Q.ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1'
LINK = Q.ROOT.parent/'g2_rolling_prefix_saved_link_2026-09-10_v1'
LIVE = Q.ROOT.parent/'g2_rolling_live_adapter_2026-09-10_v1'


def verify(kept: Any) -> Any:
    runtime,factory,state = kept['runtime'],kept['factory'],kept['state']
    output,control = state['output'],factory.controller
    read = lambda name:[json.loads(line) for line in (output/name).read_text().splitlines()]
    full,journal = read('directional_history.jsonl'),read('atomic_journal.jsonl')
    rows = [r for r in full if 'decision' in r]
    outer = json.loads((output/'POSTCOMMIT_CONSUMER_ROWS.json').read_bytes())
    lease = state['repeat_scope_guard'].reset_lease
    boundary = B.publication(factory,state,lease.recovery)
    frames = [r['scope']['frame_idx'] if 'decision' in r else r['frame'] for r in full]
    assert frames==list(range(frames[0],outer[-1]['frame_idx']+Q.E.STRIDE,Q.E.STRIDE))
    assert hashlib.sha256(ADAPTER.read_bytes()).hexdigest()==ADAPTER_SHA
    adapter = Q.module('_empty_reset_stage_adapter',ADAPTER)
    baseline = next(r for r in lease.recovery.rows if r['kind']=='new_baseline')
    BOUNDARY.verify(lease,factory,baseline,full)
    evidence = json.loads((output/'EMPTY_TAIL_LIVE_EVIDENCE.json').read_bytes())
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path[:0] = [str(Q.FINAL),str(PRIVATE),str(LINK)]
        helper = runtime.load('_empty_reset_finish_helpers',PRIVATE/'fusion.py',stack)
        S = runtime.load('_empty_reset_finish_stage1',Q.FINAL/'empty_stage1.py',stack)
        join = runtime.load('_empty_reset_finish_join',LIVE/'prepop_join.py',stack)
        with S.libraries() as lib,lib.F.session() as stage:
            function = adapter.derive(S,stage,lib,evidence,rows,baseline)
            result = function(helper.goals(runtime,stack,rows,outer),rows,control.legal,output,
                firing_rows=[],journal_rows=journal,conditional_rows=state['conditional_full_rows'],
                controller=control,factory=factory)
            saved = lib.L.check(rows,journal)
        joined = join.check(control.hidden_rolling_prepop_checks,saved,rows)
    assert result['same_live_controller_verified'] and joined['rolling_exercised']
    assert result['issued']==len(boundary['issued_frames']) and result['released']==0
    return dict(stage1=result,rolling_link=saved,rolling_join=joined,boundary=boundary,
        wait_rows_retained=len(full)-len(rows),global_J_and_publication_once=True,
        actual_factory=True,quality_gate_clear=False)
