"""後段のframes prefix差だけを既存採否証明へ束縛する。既定は介入なし。"""
from __future__ import annotations
import ast
import contextlib
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
LEGACY = VERIFY / 'g2_split_diagnostic_finisher_2026-09-08_v1/finisher.py'
TORCH = VERIFY / 'g2_finisher_torch_version_2026-09-08_v1/adapter.py'
REPAIR = VERIFY / 'g2_finisher_receipt_repair_2026-09-08_v1/adapter.py'
PROOF = VERIFY / 'g2_publication_collector_finish_2026-09-09_v1/proof.py'
RUNTIME = VERIFY / 'g2_split_runtime_evidence_adapter_2026-09-08_v1/runner.py'
MARKER = '_publication_downstream_prefix_allowed'
OWN = ('adapter.py', 'test_adapter.py', 'run_cpu.py', 'ASSET_PREFLIGHT.md')
FIXED = {LEGACY: '96541c1e7df8b555e26bd5cca742d16060ba46f06c925e675adaacb5decc0702',
    REPAIR: 'e82d1cac690f77bbc819cecad29e0de294bab8fa7b94d2167ceb3b40498692a6',
    TORCH: 'cb20535bfc3b48b3e9f9caee12e03d0e0106adb7e6e5f5e80cb62a79612e6daf',
    PROOF: 'a14a0442aeac99750654202a6cd1b9de265bdaca3cd898eb0d32f90db51fc150',
    RUNTIME: '78bdc5fdc58e2b2300ddff2215bc0276be4d810ce36bfa87aeff7505924fd93e'}
PREFIX_EXPRESSION = 'value["pre_mutation_prefix_bit_exact"] is not True'
PREFIX_REPLACEMENT = 'not ' + MARKER + '(output, report, name, value, bound)'
REPORT_EXPRESSION = 'report["streams"]["frames.jsonl"]["pre_mutation_prefix_bit_exact"]'


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'downstream_fixed_source_changed')
    return {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


def code_of(path: Path, name: str) -> CodeType:
    return next(item for item in compile(path.read_bytes(), str(path), 'exec').co_consts
                if isinstance(item, CodeType) and item.co_name == name)


def module_identity(module: Any, path: Path, names: tuple[str, ...]) -> None:
    require(type(module) is ModuleType and Path(module.__file__).resolve() == path, 'downstream_module_identity')
    for name in names:
        value = getattr(module, name, None)
        require(type(value) is FunctionType and value.__code__ == code_of(path, name)
            and value.__globals__ is vars(module), 'downstream_function_identity:' + name)


def components(runtime: Any, finisher: Any, proof: Any) -> Any:
    guards()
    module_identity(runtime, RUNTIME, ('comparisons',))
    module_identity(finisher, TORCH, ('install', 'guards'))
    module_identity(finisher.OLD, REPAIR, ('install',))
    module_identity(proof, PROOF, ('verify', 'guards', 'inputs', 'digest', 'read', 'sha'))
    require(not hasattr(finisher.OLD.F, MARKER), 'downstream_reentry')
    module_identity(finisher.OLD.F, LEGACY, ('validate_comparison', 'diagnostic_reports', 'history_save', 'finish'))
    require(Path(runtime.REFERENCE).resolve() == proof.REFERENCE and proof.C6 == 30272,
        'downstream_reference_or_boundary')
    return finisher.OLD.F


def prepare_bound(output: Path, proof: Any) -> None:
    plan = json.loads((output / 'PLAN.json').read_text())
    required = guards() | proof.guards()
    actual = plan['input_and_code_sha256']
    require(all(actual.get(path) == value for path, value in required.items()), 'downstream_prepare_guard_missing')
    require(all(sha(Path(path)) == value for path, value in required.items()), 'downstream_prepare_input_changed')


def allow(output: Path, report: dict[str, Any], name: str, value: Any,
          bound: dict[str, str], proof: Any) -> bool:
    prefix = value['pre_mutation_prefix_bit_exact']
    if name != 'frames.jsonl':
        return prefix is True
    require(type(prefix) is bool, 'downstream_prefix_boolean')
    require(type(report['first_c6']) is int and report['first_c6'] == proof.C6, 'downstream_fixed_boundary')
    prepare_bound(output, proof)
    proof.verify(output)
    receipt_path, replay_path = output / proof.RECEIPT, output / proof.REPLAY
    receipt = proof.read(receipt_path)
    require(receipt['original_report_digest'] == proof.digest(report), 'downstream_report_digest')
    require(receipt['old_prefix_bit_exact'] is prefix, 'downstream_prefix_receipt_mismatch')
    required = proof.guards() | proof.inputs(output)
    require(all(receipt['guards'].get(p) == h for p, h in required.items()), 'downstream_proof_guard_missing')
    bound.update({str(receipt_path): sha(receipt_path), str(replay_path): sha(replay_path)})
    return True


def transformed(name: str) -> CodeType:
    node = next(n for n in ast.parse(LEGACY.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    matches = []
    if name == 'validate_comparison':
        expected = ast.dump(ast.parse(PREFIX_EXPRESSION, mode='eval').body, include_attributes=False)
        for item in ast.walk(node):
            if isinstance(item, ast.BoolOp):
                matches += [(item.values, i) for i, val in enumerate(item.values)
                    if ast.dump(val, include_attributes=False) == expected]
        replacement = ast.parse(PREFIX_REPLACEMENT, mode='eval').body
    elif name == 'diagnostic_reports':
        for item in ast.walk(node):
            if isinstance(item, ast.Dict):
                matches += [(item.values, i) for i, key in enumerate(item.keys)
                    if isinstance(key, ast.Constant) and key.value == 'prefix_bit_exact'
                    and isinstance(item.values[i], ast.Constant) and item.values[i].value is True]
        replacement = ast.parse(REPORT_EXPRESSION, mode='eval').body
    else:
        raise ValueError('unsupported_downstream_transform')
    require(len(matches) == 1, 'downstream_single_ast_site')
    values, index = matches[0]
    values[index] = ast.copy_location(replacement, values[index])
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    return next(n for n in compile(tree, str(LEGACY), 'exec').co_consts if isinstance(n, CodeType))


def install(stack: contextlib.ExitStack, runtime: Any, finisher: Any, proof: Any,
            *, enabled: bool = False) -> None:
    """driver.configured前。old captured finishは元installへ任せ、二関数だけscope内で置換する。"""
    require(type(enabled) is bool, 'downstream_enabled_boolean')
    if not enabled:
        return
    module = components(runtime, finisher, proof)
    require(not hasattr(module, MARKER), 'downstream_reentry')
    names = ('validate_comparison', 'diagnostic_reports')
    originals = {name: getattr(module, name) for name in names}
    changed = {name: FunctionType(transformed(name), vars(module), name, fn.__defaults__)
               for name, fn in originals.items()}
    def permitted(output: Path, report: Any, name: str, value: Any, bound: dict[str, str]) -> bool:
        return allow(output, report, name, value, bound, proof)
    def restore() -> None:
        for name, original in originals.items():
            setattr(module, name, original)
        delattr(module, MARKER)
    setattr(module, MARKER, permitted)
    for name, function in changed.items():
        function.__kwdefaults__ = originals[name].__kwdefaults__
        function.__annotations__ = dict(originals[name].__annotations__)
        setattr(module, name, function)
    stack.callback(restore)
