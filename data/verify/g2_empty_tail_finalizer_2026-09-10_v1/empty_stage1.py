"""元Stage1の私有分岐を合成し、新kindの保存J連結だけを付加する。"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any, Iterator
import empty_journal as J

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT.parent/'g2_private_suffix_finalizer_stage1_2026-09-10_v1'
BRIDGE = ROOT.parent/'g2_rolling_stage1_bridge_2026-09-10_v1/bridge.py'
LINK = ROOT.parent/'g2_rolling_prefix_saved_link_2026-09-10_v1'
PINS = {
    BRIDGE:'d41505c000f5273a6b0d289b44a6783014d6d7b3ae385b16c494f260b68475a5',
    PRIVATE/'compat.py':'815e48df736b6971e894e4ebbba20b38355eaba11e23ed3f20b6c3aa22c9320d',
    PRIVATE/'private_journal.py':'25e6f7981115895aaeb969d1223d0f7b7dd28093c573482bffe26902bc57ede5',
    LINK/'saved_link.py':'c9bc4b1aefc56df32c00ba11cd8a95b05b66aac454973d2f70cb41970be26b2e',
    LINK/'rolling_link_fixed.py':'8c231fbb2e8f71766b072767552b4407942d7a154cfc7504ed707f9685f1b129',
    LINK/'rolling_link_journal.py':'15b20f35a146615908787213fd65b33befde3d157bf4e07b47856c4c54173e45'}


def load(name: str, path: Path) -> Any:
    J.require(hashlib.sha256(path.read_bytes()).hexdigest() == PINS[path], 'dependency_SHA')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def libraries() -> Iterator[Any]:
    names = ('private_fixed','private_generation','private_journal','rolling_link_fixed','rolling_link_journal')
    previous = {name:sys.modules.get(name) for name in names}
    try:
        bridge = load('_empty_stage1_bridge', BRIDGE)
        lib = bridge.libraries()
        sys.modules['private_fixed'], sys.modules['private_generation'] = lib.F, lib.G
        private_journal = load('_empty_private_journal', PRIVATE/'private_journal.py')
        sys.modules['private_journal'] = private_journal
        private = load('_empty_private_compat', PRIVATE/'compat.py')
        sys.modules['rolling_link_fixed'] = load('_empty_link_fixed', LINK/'rolling_link_fixed.py')
        sys.modules['rolling_link_journal'] = load('_empty_link_journal', LINK/'rolling_link_journal.py')
        link = load('_empty_saved_link', LINK/'saved_link.py')
        yield N(B=bridge,F=lib.F,G=lib.G,P=private,L=link)
    finally:
        for name, module in previous.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module
        J.require(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in PINS.items()), 'dependency_changed')


def selected(rows: Any) -> bool:
    return any(type(r.get('prepared')) is dict and r['prepared'].get('kind') == J.KIND for r in rows)


def derive(stage: Any, lib: Any, evidence: Any) -> Any:
    """元private deriveの同body合成。原pop/state/整数の本体は変えない。"""
    journal = N(**vars(stage.J))
    journal.KINDS = stage.J.KINDS+(J.KIND,)
    def created(audit: Any, token: str, item: Any) -> Any:
        return lib.G.created(stage.J, audit, token, item)
    journal.created = created
    journal.pop = lib.F.clone(stage.J.pop, created=created)
    state = N(**vars(stage.S))
    state.advance = lib.F.clone(stage.S.advance, J=journal)
    base = N(**(vars(stage)|dict(J=journal,S=state)))
    function = lib.P.derive(base)
    original_state = function.__globals__['S']
    def verify(audit: Any, legal: Any) -> Any:
        report = J.verify(audit, evidence, journal)
        value = original_state.verify(audit, legal)
        return value | dict(empty_tail=report)
    output_state = N(**(vars(original_state)|dict(verify=verify)))
    return lib.F.clone(function, S=output_state)


def run(stage: Any, lib: Any, evidence: Any, *args: Any, **kwargs: Any) -> Any:
    rows = args[1] if len(args) > 1 else kwargs['rows']
    if not selected(rows):
        J.require(not evidence, 'empty_evidence_without_new_kind')
        if lib.P.selected(rows):
            return lib.P.audit_stage1(*args, **kwargs)
        return lib.B.derive(stage.audit_stage1)(*args, **kwargs)
    journal = kwargs['journal_rows']
    linked = lib.L.check(rows, journal)
    result = derive(stage, lib, evidence)(*args, **kwargs)
    return result | dict(empty_tail_structure_verified=True, empty_tail_samecall_verified=False,
        empty_tail_saved_link=linked, runtime_finalization_allowed=False,
        original_Link_quiet_reauthorized=False, private_writer_replayed=False, new_proofs_relabelled=False)


@contextmanager
def installed(stage1: Any, *, empty_evidence: Any = None) -> Iterator[Any]:
    """原conditional moduleを外側facadeにする。非対象は既存private/rolling経路へ。"""
    with libraries() as lib:
        lib.B.verify(stage1, lib)
        originals = [(m,dict(vars(m))) for m in (stage1,stage1.J,stage1.S)]
        def audit(*args: Any, **kwargs: Any) -> Any:
            evidence = kwargs.pop('empty_evidence', empty_evidence)
            return run(stage1, lib, evidence, *args, **kwargs)
        def evaluate(*args: Any, **kwargs: Any) -> Any:
            rows = args[1] if len(args) > 1 else kwargs['rows']
            J.require(not selected(rows), 'empty_world_and_samecall_required')
            J.require(not kwargs.pop('empty_evidence', empty_evidence), 'empty_evidence_without_new_kind')
            if lib.P.selected(rows):
                return lib.P.evaluate(*args, **kwargs)
            return stage1.evaluate(*args, **kwargs)
        facade = N(**(vars(stage1)|dict(audit_stage1=audit,evaluate=evaluate)))
        try:
            yield facade
        finally:
            lib.B.verify(stage1, lib)
            J.require(all(set(vars(m)) == set(old) and all(vars(m)[k] is v for k,v in old.items())
                for m,old in originals), 'original_globals_changed')


def audit_stage1(goals: Any, rows: Any, legal: Any, output: Any, *, firing_rows: Any,
                 journal_rows: Any, conditional_rows: Any, controller: Any = None,
                 factory: Any = None, empty_evidence: Any = None) -> dict[str, Any]:
    with libraries() as lib, lib.F.session() as stage:
        with installed(stage, empty_evidence=empty_evidence) as facade:
            return facade.audit_stage1(goals,rows,legal,output,firing_rows=firing_rows,
                journal_rows=journal_rows,conditional_rows=conditional_rows,controller=controller,factory=factory)
