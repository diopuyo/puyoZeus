"""既存proofの正本を保ち、同call色留保だけを私有比較へ追加する。"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
PROJECT, VERIFY = ROOT.parents[2], ROOT.parent
PROOF = VERIFY / 'g2_publication_collector_finish_2026-09-09_v1/proof.py'
PALETTE = VERIFY / 'g2_palette_evidence_veto_2026-09-09_v1/adapter.py'
METADATA = VERIFY / 'g2_collector_metadata_capture_2026-09-09_v1/observer.py'
REPLAY_SOURCE = VERIFY / 'g2_bounded_publication_runtime_2026-09-09_v1/analyze_metadata.py'
JOURNAL = VERIFY / 'g2_atomic_journal_capture_2026-09-09_v1/observer.py'
HISTORY_TEST = VERIFY / 'g2_publication_downstream_finisher_2026-09-09_v1/test_adapter.py'
BASELINE = VERIFY / 'video38_atomic_journal_live_2026-09-09_v1'
FIXED = {PROOF: 'a14a0442aeac99750654202a6cd1b9de265bdaca3cd898eb0d32f90db51fc150',
    PALETTE: '33ab8251d5b930558cc9b1a4a8b95b1e663daa4feb70e29dc438aebd8d5c05fc',
    METADATA: '98b68c0f6a523bc1c756f359e02fc3f345a4bfcc234ea0ba9262610daa9b7e1a',
    REPLAY_SOURCE: 'bf974281f8fdf5990a18522789e01ec63131c412cd9bdfb9e80e9a846ea81fda',
    JOURNAL: 'b0d956a4fef51846a7baa2cee95a3679887bd2e3e9e620462e8375dc238d830c',
    HISTORY_TEST: '3b1a6c620f64ce98fc74a284d915a44b1b647bdd91a62e964848e5b874a92e26',
    BASELINE / 'COMPLETE': '97de82f6a8e5b517cb560c778ff8359b8e8057fc1c729b76d9f9ca41d391e26a'}
OWN = ('adapter.py', 'evidence.py', 'collector_replay.py', 'cpu_fixture.py', 'history_cpu.py',
       'test_adapter.py', 'run_cpu.py', 'CONTRACT.md')
REQUIRED = frozenset(('PUBLICATION_PREFIX_RECEIPT.json', 'PUBLICATION_PREFIX_REPLAY.json'))
EXTRA = ('palette_evidence_veto.jsonl', 'PALETTE_EVIDENCE_VETO.json', 'atomic_journal.jsonl',
         'collector_metadata.jsonl', 'COLLECTOR_METADATA_STATUS.json')
FIRST, LAST, STRIDE, FPS, C6 = 29052, 36298, 2, 60, 30272
SIDES = ('1P', '2P')
SOURCE_ID = 'sha256:b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pairs(values: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        require(key not in result, 'palette_proof_duplicate_json_key')
        result[key] = value
    return result


def parse(data: str | bytes) -> Any:
    return json.loads(data, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite_json:' + x)))


def read(path: Path) -> Any:
    return parse(path.read_bytes())


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def equal(left: Any, right: Any) -> bool:
    return digest(left) == digest(right)


def lines(path: Path) -> Iterator[Any]:
    with path.open('rb') as stream:
        for line in stream:
            yield parse(line)


def load(path: Path, suffix: str) -> Any:
    alias = '_palette_private_proof_' + suffix
    require(alias not in sys.modules, 'palette_proof_alias_collision')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[alias]
    return module


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'palette_proof_fixed_source')
    baseline = read(BASELINE / 'COMPLETE')
    fixed_inputs = {str(BASELINE / n): baseline['sha256'][n] for n in ('frames.jsonl', 'PLAN.json')}
    require(all(sha(Path(p)) == h for p, h in fixed_inputs.items()), 'palette_proof_baseline_changed')
    return {str(p): h for p, h in FIXED.items()} | fixed_inputs | {str(ROOT / n): sha(ROOT / n) for n in OWN}


def source_functions(path: Path, names: tuple[str, ...], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in selected} == set(names), 'palette_proof_function_missing')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def dependencies() -> Any:
    e, c = load(ROOT / 'evidence.py', 'evidence'), load(ROOT / 'collector_replay.py', 'collector')
    e.A = c.A = sys.modules.get(__name__) or SimpleNamespace(**globals())
    return SimpleNamespace(e=e, c=c, palette=load(PALETTE, 'palette'), metadata=load(METADATA, 'metadata'),
        journal=load(JOURNAL, 'journal'))


def identity(proof: Any) -> None:
    require(Path(proof.__file__).resolve() == PROOF and sha(PROOF) == FIXED[PROOF], 'palette_proof_module')
    compiled = compile(PROOF.read_bytes(), str(PROOF), 'exec')
    codes = {v.co_name: v for v in compiled.co_consts if hasattr(v, 'co_name')}
    for name in ('prove', 'report_contract', 'replay', 'verify', 'inputs', 'guards', 'digest', 'read', 'sha'):
        fn = getattr(proof, name)
        require(type(fn) is FunctionType and fn.__code__ == codes[name]
            and fn.__globals__ is vars(proof), 'palette_proof_function:' + name)


def cloned_prove(proof: Any, dep: Any) -> Any:
    active: dict[str, Any] = {}
    def report(output: Path, value: Any) -> None:
        require(not active, 'palette_proof_reentry')
        active.update(dep.e.audit(proof, dep, output, value))
    def replay(output: Path, collector: Any) -> Any:
        require(active.get('output') == str(output.resolve()), 'palette_proof_audit_missing')
        return dep.c.replay(proof, dep, output, collector, active)
    namespace = dict(vars(proof))
    namespace.update(guards=lambda: proof.guards() | guards(), report_contract=report, replay=replay,
        inputs=lambda output: proof.inputs(output) | {str(output / n): sha(output / n) for n in EXTRA})
    original = proof.prove
    private = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    def invoke(output: Path, value: Any, state: Any) -> None:
        require(not active, 'palette_proof_reentry')
        try:
            private(output, value, state)
        finally:
            active.clear()
    return invoke


def install(stack: Any, proof: Any, *, enabled: bool = False) -> None:
    require(type(enabled) is bool, 'palette_proof_enable_type')
    if not enabled:
        return
    guards()
    identity(proof)
    original = proof.prove
    replacement = cloned_prove(proof, dependencies())
    stack.callback(setattr, proof, 'prove', original)
    proof.prove = replacement


def verify(output: Path) -> None:
    before = guards()
    proof = load(PROOF, 'verify')
    proof.verify(output)
    receipt, replay = read(output / proof.RECEIPT), read(output / proof.REPLAY)
    require(all(receipt['guards'].get(p) == h for p, h in before.items()), 'palette_proof_saved_guards')
    require(all(str(output / n) in receipt['guards'] for n in EXTRA), 'palette_proof_saved_inputs')
    value = replay['palette_compatibility']
    require(value['same_call_veto_bound'] is True and value['virtual_lane_not_physical_truth'] is True
        and value['virtual_prefix_core_equal'] is True and value['metadata_21_columns_equal'] is True,
        'palette_proof_saved_contract')
    require(replay['downstream_core_equal'] is True and value['quality_gate_clear'] is False
        and value['accounting_permission'] is False and value['physical_truth_certified'] is False,
        'palette_proof_saved_permissions')


def finish(state: dict[str, Any]) -> None:
    verify(Path(state['output']))
