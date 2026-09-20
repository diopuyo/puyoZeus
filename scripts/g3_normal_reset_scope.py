"""検収済み早期reset保留を、正常M1の動的side観測へ限定再利用する。"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable
from scripts import g3_reset_scope as R

SOURCE_SHA = '91673e42d483386ef709b44d2d1fa2537d76e2aae0b370c0c14daede7e944da9'
FLAGS_SHA = 'f114f1cf2348a509cb90536ce5d592a771a8c2c22453a96185a2cdf2fa3c1e83'
FLAGS_MARKER = "    require(local['sm'] is sm and sm.context.frame_idx == scope['frame_idx'], 'SM_clock')\n"


def install_flags(stack: Any, module: Any, replace: Callable) -> dict:
    """同step resetだけ保留し、原票との照合用baseはそのまま保持する。"""
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != FLAGS_SHA:
        raise ValueError('normal_flags_source_changed')
    original = module.capture
    if original.__globals__ is not vars(module) or original.__name__ != 'capture':
        raise ValueError('normal_flags_function_owner')
    source = inspect.getsource(original)
    if source.count(FLAGS_MARKER) != 1:
        raise ValueError('normal_flags_marker')
    insertion = R.INSERT.split('        return dict(frame=', 1)[0]
    insertion = insertion.replace('C.require', 'require').replace('SIDE', 'side')
    insertion += "        return base | dict(hold_reason='generation_changed')\n"
    transformed = source.replace(FLAGS_MARKER, insertion + FLAGS_MARKER)
    namespace: dict[str, Any] = {}
    exec(compile(transformed, str(Path(__file__).resolve()), 'exec'), vars(module), namespace)
    replace(stack, module, 'capture', namespace['capture'])
    return dict(original_sha256=FLAGS_SHA, generated_source=transformed,
        generated_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
        preserved_base=True, quality_gate_clear=False)


def generated(module: Any) -> tuple[Callable, dict]:
    """原globalsとside引数を保ち、未登録resetも観測不足HOLDへ分類する。"""
    path = Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('normal_reset_source_changed')
    original = module.capture
    if original.__globals__ is not vars(module) or original.__name__ != 'capture':
        raise ValueError('normal_reset_function_owner')
    source = inspect.getsource(original)
    if source.count(R.MARKER) != 1:
        raise ValueError('normal_reset_marker')
    insertion = R.INSERT.replace('SIDE', 'side')
    if 'SIDE' in insertion:
        raise ValueError('normal_reset_fixed_side_remaining')
    transformed = source.replace(R.MARKER, insertion + R.MARKER)
    namespace: dict[str, Any] = {}
    exec(compile(transformed, str(Path(__file__).resolve()), 'exec'), vars(module), namespace)
    value = namespace['capture']
    assert module.capture is original and value.__globals__ is vars(module)
    return value, dict(original_sha256=SOURCE_SHA, generated_source=transformed,
        generated_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
        dynamic_side=True, unregistered_reset_policy='HOLD_no_basis_permission', quality_gate_clear=False)


def install(stack: Any, module: Any, replace: Callable) -> dict:
    """正常module固有の所有stackへ設置し、元captureを復元する。"""
    value, receipt = generated(module)
    replace(stack, module, 'capture', value)
    return receipt
