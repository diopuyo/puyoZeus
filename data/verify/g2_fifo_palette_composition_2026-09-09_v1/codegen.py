"""固定の原builder/validatorを実行し、モデルなしでOFF/ON二codeを導出する。"""
from __future__ import annotations
import ast
import builtins
from contextlib import ExitStack
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
VERIFY = ROOT.parent
SOURCE = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py'
PENDING = PROJECT / 'scripts/diagnose_video38_c6_pending_commit_shadow_v1.py'
T2 = VERIFY / 'g2_t2_fresh_landing_guard_2026-09-09_v1/adapter.py'
FIFO = VERIFY / 'g2_landing_fifo_pair_binding_2026-09-09_v1/adapter.py'
GRACE = VERIFY / 'g2_resolved_grace_guard_shadow_2026-09-08_v1/shadow.py'
JOURNAL = VERIFY / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
FIXED = {SOURCE: '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02',
    PENDING: '563fb383e9c0ceb804c3950522a67c77460a69f5e483a0c09c9becdd2f6cbab0',
    T2: 'bd053d7070656cb652e8d84a77053f2d07fc25237fc9fa61c0c64fc28309a8a1',
    FIFO: '72ad4a4b541e5148a6a10dcfdcae2af9e4ad1827364e643172d24febaa81fa7b',
    GRACE: 'f20028f16c980825ee435ff57df649d1527b740c04861de0e621d62011af7561',
    JOURNAL: 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c'}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(path) == value for path, value in FIXED.items()), 'composition_source_changed')
    return {str(path): value for path, value in FIXED.items()}


def module(path: Path, *, imports: Any = None) -> Any:
    """ファイルを私有moduleへロードし、共有sys.modulesを差し替えない。"""
    require(path in FIXED and sha(path) == FIXED[path], 'composition_module_source')
    value = ModuleType('_fifo_palette_private_' + path.stem)
    value.__file__ = str(path)
    value.__dict__['__builtins__'] = dict(vars(builtins), __import__=imports or builtins.__import__)
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), value.__dict__)
    return value


def pending_module() -> Any:
    names = {'_is_c6_block', '_hook_statement', '_build_transformed_step'}
    tree = ast.parse(PENDING.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'composition_pending_functions')
    value = ModuleType('_fifo_palette_pending_compile_only')
    value.__dict__.update(ast=ast, inspect=inspect, textwrap=textwrap,
                          functools=functools, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PENDING), 'exec'), value.__dict__)
    return value


def patch(stack: Any, owner: Any, name: str, value: Any) -> None:
    old = getattr(owner, name)
    stack.callback(setattr, owner, name, old)
    setattr(owner, name, value)


def original_step() -> Any:
    code = compile(SOURCE.read_text(), str(SOURCE), 'exec', dont_inherit=True)
    cls = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'RecognitionPipeline')
    step = next(c for c in cls.co_consts if isinstance(c, CodeType) and c.co_name == '_step_side')
    return FunctionType(step, {'__name__': '_fifo_palette_unexecuted_step'}, '_step_side')


def runtime_for_compile() -> tuple[Any, Any]:
    pending = pending_module()
    def imports(name: str, globals: Any = None, locals: Any = None,
                fromlist: Any = (), level: int = 0) -> Any:
        if name == 'scripts' and tuple(fromlist) == ('diagnose_video38_c6_pending_commit_shadow_v1',) and level == 0:
            return SimpleNamespace(diagnose_video38_c6_pending_commit_shadow_v1=pending)
        return builtins.__import__(name, globals, locals, fromlist, level)
    def load(name: str, path: Path) -> Any:
        require(path == GRACE, 'composition_unexpected_compile_module')
        return module(path, imports=imports)
    prior = SimpleNamespace(load=load)
    runtime = SimpleNamespace(pending=pending, A=SimpleNamespace(prior=prior), base=SimpleNamespace(patch=patch))
    return runtime, load


def code_record(grace: Any, code: CodeType) -> dict[str, Any]:
    semantic = json.dumps(canonical(grace.code_key(code)), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'bytecode_sha256': hashlib.sha256(code.co_code).hexdigest(),
        'semantic_sha256': hashlib.sha256(semantic.encode()).hexdigest(),
        'filename': str(Path(code.co_filename).resolve()), 'firstlineno': code.co_firstlineno}


def canonical(value: Any) -> Any:
    if type(value) is bytes:
        return {'bytes': value.hex()}
    if type(value) in (tuple, list):
        return [canonical(item) for item in value]
    if type(value) is frozenset:
        return {'frozenset': sorted((canonical(item) for item in value), key=repr)}
    require(type(value) in (str, int, float, bool, type(None)), 'composition_code_constant_type')
    return value


def derive_lane(t2: Any, fifo: Any, enabled: bool) -> dict[str, Any]:
    runtime, old_load = runtime_for_compile()
    original = runtime.pending._build_transformed_step
    hook = lambda *args: None
    with ExitStack() as stack:
        t2.install(stack, runtime, enabled=True)
        fifo.install(stack, runtime, t2, enabled=enabled)
        pending, receipt = runtime.pending._build_transformed_step(original_step(), hook)
        grace = runtime.A.prior.load('fixed_grace', GRACE)
        guard = SimpleNamespace(capture=lambda *args: None, advance=lambda *args: None, receipt={})
        resolved = grace.compile_step(stack, pending, guard)
        require(pending.__globals__['__c6_pending_shadow_hook'] is hook, 'composition_hook_changed')
        result = {'pending': code_record(grace, pending.__code__),
            'resolved': code_record(grace, resolved.__code__), 'c6_receipt': receipt,
            'grace_receipt': guard.receipt, 'enabled': enabled}
    require(runtime.pending._build_transformed_step is original and runtime.A.prior.load is old_load,
            'composition_compile_restore')
    require('__resolved_capture' not in pending.__globals__ and '__resolved_advance' not in pending.__globals__,
            'composition_helper_restore')
    return result


def derive() -> dict[str, Any]:
    before = guards()
    t2, fifo, journal = module(T2), module(FIFO), module(JOURNAL)
    off, on = derive_lane(t2, fifo, False), derive_lane(t2, fifo, True)
    old = {off[name]['bytecode_sha256'] for name in ('pending', 'resolved')}
    require(old == journal.CODE_HASHES and len(old) == 2, 'composition_off_not_original_journal')
    require(len({on[name]['bytecode_sha256'] for name in ('pending', 'resolved')}) == 2,
            'composition_new_two_codes')
    require(all(off[name]['bytecode_sha256'] != on[name]['bytecode_sha256']
                for name in ('pending', 'resolved')), 'composition_fifo_not_applied')
    require(before == guards(), 'composition_derivation_source_changed')
    return {'schema': 'fifo-palette-code-derivation/v1', 'off': off, 'on': on, 'source_sha256': before,
        'recognition_executed': False, 'physical_certified': False, 'quality_gate_clear': False}
