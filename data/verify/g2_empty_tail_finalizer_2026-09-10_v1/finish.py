"""元Stage1/J、world/PB、実成功参照、rolling保存を全て満たした時だけ閉鎖する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT.parent
PRIVATE=VERIFY/'g2_private_suffix_fusion_2026-09-10_v1'
LINK=VERIFY/'g2_rolling_prefix_saved_link_2026-09-10_v1'
LIVE=VERIFY/'g2_rolling_live_adapter_2026-09-10_v1'


def finish(kept: Any) -> None:
    runtime,factory,state=kept['runtime'],kept['factory'],kept['state']
    control,output=factory.controller,state['output']
    with ExitStack() as stack:
        previous=list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path[:0]=[str(ROOT),str(PRIVATE),str(LINK)]
        helper=runtime.load('_empty_finish_old_helpers',PRIVATE/'fusion.py',stack)
        stage1=runtime.load('_empty_finish_stage1',ROOT/'empty_stage1.py',stack)
        world=runtime.load('_empty_finish_world',ROOT/'empty_world.py',stack)
        linked=runtime.load('_empty_finish_rolling_link',LINK/'saved_link.py',stack)
        join=runtime.load('_empty_finish_rolling_join',LIVE/'prepop_join.py',stack)
        step=runtime.load('_empty_finish_original_step',PRIVATE/'samecall.py',stack).journal_step
        rows,journal=(helper.read(output/name) for name in ('directional_history.jsonl','atomic_journal.jsonl'))
        outer=helper.read(output/'POSTCOMMIT_CONSUMER_ROWS.json')
        with world.loaded() as original_world:
            with original_world.libraries().G.libraries(): pass
        structure=stage1.audit_stage1(helper.goals(runtime,stack,rows,outer),rows,control.legal,output,
            firing_rows=[],journal_rows=journal,conditional_rows=state['conditional_full_rows'],
            controller=control,factory=factory,empty_evidence=helper.read(output/'EMPTY_TAIL_LIVE_EVIDENCE.json'))
        checked=world.verify_world(history_rows=rows,journal_rows=journal,
            conditional_rows=state['conditional_full_rows'],hidden_events=control.hidden_current_events,
            hidden_history=control.hidden_history_rows,hidden_lifetime=control.hidden_lifetime_rows,
            outer_rows=outer,controller=control,factory=factory)
        parts=kept['empty_parts']
        observed=parts.completion.verify(factory,parts.C,journal,rows,step)
        saved=linked.check(rows,journal)
        joined=join.check(control.hidden_rolling_prepop_checks,saved,rows)
        assert structure['empty_tail_structure_verified'] and structure['same_live_controller_verified']
        assert checked['actual_live_scope_verified']
        assert checked['world_PB_verified'] and observed['empty_commit_samecall_verified']
        assert joined['rolling_exercised'] and checked['original_capture_calls']['producer_call_coverage_verified']
        runtime.write(output/'EMPTY_FUSED_RUNTIME.json',dict(stage1=structure,world=checked,
            samecall=observed,rolling_link=saved,rolling_join=joined,runtime_finalization_allowed=True,
            actual_factory=True,artificial_raw_next_clock=True,original_live_constructor=False,
            physical_certified=False,quality_gate_clear=False,GPU_GO=False))
