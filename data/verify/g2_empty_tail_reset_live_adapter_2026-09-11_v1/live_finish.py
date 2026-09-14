"""実goals/firingを保持した多scope最終検収。計算閉鎖をG2合格へ昇格しない。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from types import CodeType
from typing import Any
import run_qualification as Q
import retired_completion as RETIRED
import runtime_world as WORLD
import stage_boundary as BOUNDARY
import runtime_stage as STAGE
import live_boundary as PUBLICATION

KEY = 'live_empty_reset_finalizer'
PRIVATE_SHA = '24e403ddc4faf2dc43628a834af09ac6463ede34aa580c647df6486c29f5f260'


def private_samecall(factory: Any, lease: Any, journal: Any, rows: Any, evidence: Any) -> Any:
    proof = lease.empty_evidence
    module = sys.modules[type(factory.controller.private_suffix_completion_rows[0]).__module__]
    path = Q.ROOT.parent/'g2_private_suffix_live_adapter_2026-09-10_v1/completion.py'
    raw = path.read_bytes()
    assert Path(module.__file__).resolve()==path and hashlib.sha256(raw).hexdigest()==PRIVATE_SHA
    expected = next(c for c in compile(raw,str(path),'exec',dont_inherit=True).co_consts
                    if isinstance(c,CodeType) and c.co_name=='verify')
    assert module.verify.__code__==expected and module.verify.__globals__ is vars(module)
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
    original = "control.history[value['scope'][-1]] is binding and current.scope == state.scope"
    matches = [n for n in ast.walk(function) if isinstance(n,ast.Assert) and ast.unparse(n.test)==original]
    assert len(matches)==1
    matches[0].test = ast.parse("retired(lease,factory,binding,current,state,(receipt.frame,value['scope'][-1]))",mode='eval').body
    namespace = dict(vars(module),retired=RETIRED.retired_owner,lease=lease)
    exec(compile(ast.fix_missing_locations(ast.Module([function],[])),str(path),'exec'),namespace)
    return namespace['verify'](factory,evidence,journal,rows,proof.step)


def read(output: Path, name: str) -> Any:
    text = (output/name).read_text()
    return [json.loads(line) for line in text.splitlines()] if name.endswith('.jsonl') else json.loads(text)


def prepared(factory: Any, state: Any, lease: Any, evidence: Any) -> Any:
    output = state['output']
    full,journal = read(output,'directional_history.jsonl'),read(output,'atomic_journal.jsonl')
    rows = [r for r in full if 'decision' in r]
    assert all('decision' in r or r.get('kind')=='reset_baseline_wait' for r in full)
    retired = RETIRED.verify(lease,factory)
    world = WORLD.verify(factory,state,lease)
    private = None
    if factory.controller.private_suffix_completion_rows:
        private = private_samecall(factory,lease,journal,rows,evidence)
    boundary = PUBLICATION.check(state['postcommit_publication_consumer'].rows,full,journal,lease.recovery.rows)
    return dict(retired=retired,world=world,private=private,boundary=boundary,quality_gate_clear=False)


def stage(goals: Any, supplied: Any, legal: Any, output: Path, state: Any, prepared_result: Any,
          *, firing_rows: Any) -> Any:
    factory = state['private_suffix_factory']
    lease = state['repeat_scope_guard'].reset_lease
    full,journal = read(output,'directional_history.jsonl'),read(output,'atomic_journal.jsonl')
    rows = [row for row in full if 'decision' in row]
    assert supplied==full or supplied==rows
    frames = [r['scope']['frame_idx'] if 'decision' in r else r['frame'] for r in full]
    assert frames==list(range(frames[0],frames[-1]+Q.E.STRIDE,Q.E.STRIDE))
    baseline, = [r for r in lease.recovery.rows if r['kind']=='new_baseline']
    BOUNDARY.verify(lease,factory,baseline,full)
    assert hashlib.sha256(STAGE.ADAPTER.read_bytes()).hexdigest()==STAGE.ADAPTER_SHA
    adapter = Q.module('_live_reset_stage_adapter',STAGE.ADAPTER)
    evidence = read(output,'EMPTY_TAIL_LIVE_EVIDENCE.json')
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path[:0] = [str(Q.FINAL),str(STAGE.PRIVATE),str(STAGE.LINK)]
        S = Q.module('_live_reset_finish_stage1',Q.FINAL/'empty_stage1.py')
        join = Q.module('_live_reset_finish_join',STAGE.LIVE/'prepop_join.py')
        with S.libraries() as lib,lib.F.session() as original:
            function = adapter.derive(S,original,lib,evidence,rows,baseline)
            result = function(goals,rows,legal,output,
                firing_rows=firing_rows,journal_rows=journal,
                conditional_rows=state['conditional_full_rows'],controller=factory.controller,factory=factory)
            saved = lib.L.check(rows,journal)
        joined = join.check(factory.controller.hidden_rolling_prepop_checks,saved,rows)
    assert result['same_live_controller_verified'] and joined['rolling_exercised']
    assert result['issued']==len(prepared_result['boundary']['issued_frames']) and result['released']==0
    assert prepared_result['world']['world']['world_PB_verified']
    return result|dict(reset_validation=prepared_result,rolling_prepop_join=joined,
        reset_wait_rows=len(full)-len(rows),actual_goals_passthrough=True,actual_firing_rows_passthrough=True,
        runtime_finalization_allowed=True,physical_certified=False,production_permission=False,quality_gate_clear=False)
