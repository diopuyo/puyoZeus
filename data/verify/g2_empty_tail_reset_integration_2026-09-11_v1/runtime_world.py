"""全原capture検査を保ち、旧scopeは実reset保存権限、新scopeは現役所有へ結ぶ。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from types import CodeType, FunctionType, SimpleNamespace as N
from typing import Any
import empty_reset as E
import run_qualification as Q
import retired_completion as R

PINS = {
    'g2_conditional_finalizer_world_2026-09-10_v1/world_pb.py':'a6197e2a6897b0461cef7fed5d4cedcd61c0df73a6c2b9915a80e432085698d1',
    'g2_conditional_current_call_join_2026-09-10_v1/call_join.py':'b3f2ace775913fe89887c8a7cf091f4dc97c6634883030f03a0498bd69fa9b14'}


def clone(function: Any, **changed: Any) -> Any:
    value = FunctionType(function.__code__,dict(function.__globals__,**changed),
        function.__name__,function.__defaults__,function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def rewrite(function: Any, replacements: Any, authority: Any) -> Any:
    path = Path(function.__globals__['__file__']).resolve()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PINS[str(path.relative_to(E.VERIFY))]
    compiled = compile(raw,str(path),'exec',dont_inherit=True)
    expected = next(c for c in compiled.co_consts if isinstance(c,CodeType) and c.co_name==function.__name__)
    assert function.__code__==expected and function.__closure__ is None
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function.__name__)
    found = set()
    for index,item in enumerate(node.body):
        text = ast.unparse(item)
        if text in replacements:
            assert text not in found
            node.body[index] = ast.parse(replacements[text]).body[0]
            found.add(text)
    assert found==set(replacements)
    namespace = dict(function.__globals__,reset_scope_authority=authority)
    exec(compile(ast.fix_missing_locations(ast.Module([node],[])),str(path),'exec'),namespace)
    return namespace[function.__name__]


def scope_authority(lease: Any, counts: Any, factory: Any, supplied: Any, frame: int) -> None:
    assert type(frame) is int and type(lease) is E.Lease
    proof, recovery = lease.empty_evidence,lease.recovery
    assert factory is proof.factory is recovery.factory and factory.controller is recovery.control
    if frame <= proof.frame:
        old = lease.archive.binding
        assert tuple(supplied)==tuple(old.scope), 'retired_full_scope'
        R.retired_owner(lease,factory,old,old.owner.state,old.owner.state,(frame,E.SIDE))
        counts['retired'] += 1
    else:
        baseline = [r for r in recovery.rows if r['kind']=='new_baseline']
        assert len(baseline)==1 and frame>=baseline[0]['first_owned_frame'], 'unowned_reset_window'
        binding = factory.controller.history[E.SIDE]
        assert binding is lease.empty_new_binding and binding.owner is lease.empty_new_owner
        assert tuple(supplied)==binding.scope==lease.new_scope==recovery.evidence.scope(factory,recovery.pipe)
        counts['active'] += 1


def adapted(Wrap: Any, old: Any, lease: Any, counts: Any) -> Any:
    W = old.libraries()
    authority = lambda f,s,t:scope_authority(lease,counts,f,s,t)
    scoped = rewrite(W.B.scope,{
        "equal(lib, supplied, lib.E.scope(factory, pipe), 'actual_live_scope7')":"reset_scope_authority(factory, supplied, value.frame)",
        "equal(lib, supplied, factory.controller.history[supplied[-1]].scope, 'live_binding_scope')":"pass"},authority)
    B = N(**(vars(W.B)|dict(scope=scoped)))
    api = N(**(vars(W.A)|dict(verify_probabilities=clone(W.A.verify_probabilities,B=B))))
    def load(alias: str, path: Path) -> Any:
        module = W.G.L.load(alias,path)
        if alias != '_world_original_capture_join': return module
        owner = rewrite(module.actual_owner,{
            "B.equal(lib, scope, factory.controller.history[side].scope, 'capture_actual_owned_scope')":
            "reset_scope_authority(factory, scope, row['frame'])"},authority)
        return N(**(vars(module)|dict(verify=clone(module.verify,actual_owner=owner))))
    G = N(**(vars(W.G)|dict(L=N(**(vars(W.G.L)|dict(load=load))))))
    scoped_world = N(**(vars(W)|dict(A=api,G=G)))
    proxy = N(**(vars(old)|dict(libraries=lambda:scoped_world,
        verify_world=clone(old.verify_world,libraries=lambda:scoped_world))))
    return Wrap.adapted(proxy)


def verify(factory: Any, state: Any, lease: Any) -> Any:
    control, output = factory.controller,state['output']
    read = lambda name:[json.loads(line) for line in (output/name).read_text().splitlines()]
    full,journal = read('directional_history.jsonl'),read('atomic_journal.jsonl')
    rows = [r for r in full if 'decision' in r]
    assert all('decision' in r or r.get('kind')=='reset_baseline_wait' for r in full)
    counts = dict(retired=0,active=0)
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__,slice(None),previous)
        sys.path.insert(0,str(Q.FINAL))
        world = Q.module('_empty_reset_runtime_world_original',Q.FINAL/'empty_world.py')
        with world.loaded() as old:
            result = adapted(world,old,lease,counts)(history_rows=rows,journal_rows=journal,
                conditional_rows=state['conditional_full_rows'],hidden_events=control.hidden_current_events,
                hidden_history=control.hidden_history_rows,hidden_lifetime=control.hidden_lifetime_rows,
                outer_rows=state['postcommit_publication_consumer'].rows,controller=control,factory=factory)
    assert result['world_PB_verified'] and result['actual_live_scope_verified']
    assert counts['retired'] and counts['active']
    return dict(world=result,scope_authority_calls=counts,current_history_replaced=False,
        actual_factory=True,multi_scope_finalizer_verified=False,quality_gate_clear=False)
