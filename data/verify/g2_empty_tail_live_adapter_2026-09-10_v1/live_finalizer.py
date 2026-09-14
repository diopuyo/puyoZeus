"""新kind時だけ元終了処理内で構造・world・当時実証拠をAND結合する。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent
FINAL=ROOT.parent/'g2_empty_tail_finalizer_2026-09-10_v1'
KIND='hidden_empty_tail_next_history/v1'


def audit(old: Any, goals: Any, rows: Any, legal: Any, output: Any, *, factory: Any,
          parts: Any, firing_rows: Any, conditional_rows: Any, original: Any=None) -> Any:
    if not any(r['prepared'] and r['prepared']['kind']==KIND for r in rows):
        assert not factory.controller.empty_completion_rows
        return old(goals,rows,legal,output,factory=factory,parts=parts,firing_rows=firing_rows,
            conditional_rows=conditional_rows,original=original)
    control=factory.controller
    refs=control.empty_runtime_references
    assert refs['installed'] and refs['closed'] and refs['restored']
    with ExitStack() as stack:
        previous=list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path.insert(0,str(FINAL))
        helper=sys.modules['_private_live_finalizer']
        stage1=helper.load(stack,'_empty_live_stage1',FINAL/'empty_stage1.py')
        world=helper.load(stack,'_empty_live_world',FINAL/'empty_world.py')
        step=helper.load(stack,'_empty_live_step',helper.STEP).journal_step
        journal=helper.read(output/'atomic_journal.jsonl')
        with world.loaded() as old_world:
            with old_world.libraries().G.libraries(): pass
        structure=stage1.audit_stage1(goals,rows,legal,output,firing_rows=firing_rows,journal_rows=journal,
            conditional_rows=conditional_rows,controller=control,factory=factory,
            empty_evidence=helper.read(output/'EMPTY_TAIL_LIVE_EVIDENCE.json'))
        checked=world.verify_world(history_rows=rows,journal_rows=journal,conditional_rows=conditional_rows,
            hidden_events=control.hidden_current_events,hidden_history=control.hidden_history_rows,
            hidden_lifetime=control.hidden_lifetime_rows,outer_rows=helper.read(output/'POSTCOMMIT_CONSUMER_ROWS.json'),
            controller=control,factory=factory)
        actual=control.empty_completion_parts
        observed=actual.module.verify(factory,actual.C,journal,rows,step)
        if control.private_suffix_completion_rows:
            parts.completion.verify(factory,parts.evidence,journal,rows,step)
        assert structure['empty_tail_structure_verified'] and structure['same_live_controller_verified']
        assert checked['world_PB_verified'] and checked['actual_live_scope_verified']
        assert observed['empty_commit_samecall_verified']
        return structure|dict(conditional_world=checked,empty_samecall=observed,
            runtime_finalization_allowed=True,world_PB_verified=True,conditional_current_counted_as_integer=False,
            physical_certified=False,production_permission=False,quality_gate_clear=False)
