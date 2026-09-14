"""渡された凍結部品の内容・code・相互参照だけを検査する。"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from functools import _lru_cache_wrapper
from pathlib import Path
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FIXED = {
    'C': ('g2_historical_completion_runtime_2026-09-09_v1/history_controller.py', '15880b3310207ca3119c29c41272fdc20a63a729812541c3352c1c47f9a839ef'),
    'V': ('g2_historical_completion_runtime_2026-09-09_v1/history_provider.py', 'fce0ac1ccfa8869a9bb71dcc5b4bc43f636ce7f6060b5321eb9750c46266447c'),
    'H': ('g2_historical_completion_runtime_2026-09-09_v1/history_state.py', '9d66ab677dea73d596fc33623d67fbcbd111ba92e86147de5dc5b61c63392114'),
    'T': ('g2_normal_completion_transaction_2026-09-09_v1/transaction.py', 'a1d67b6efcd21830109470a95da725a26beb21ddf5c5b6451b598a43b7dac553'),
    'O': ('g2_directional_next_enqueue_2026-09-09_v2/occurrence.py', 'b71a992a94d5231278242ac14b73dab725786e713f543956a33a9f40179b86dc'),
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def original_function(value: FunctionType) -> FunctionType:
    if not hasattr(value, '__wrapped__'):
        return value
    inner = value.__wrapped__
    expected = contextmanager(inner)
    require(value.__code__ == expected.__code__ and value.__globals__ is expected.__globals__
            and len(value.__closure__ or ()) == 1 and value.__closure__[0].cell_contents is inner,
            'handoff_contextmanager_wrapper')
    return inner


def code_members(namespace: Any, code: CodeType, globals_dict: dict[str, Any]) -> None:
    for child in code.co_consts:
        if not isinstance(child, CodeType):
            continue
        value = vars(namespace).get(child.co_name)
        if type(value) is _lru_cache_wrapper:
            require(child.co_name == 'original_method' and value.cache_parameters() == {'maxsize': 2, 'typed': False},
                    'handoff_cache_wrapper')
            value = value.__wrapped__
        if isinstance(value, type):
            code_members(value, child, globals_dict)
        elif type(value) is FunctionType:
            value = original_function(value)
            require(value.__globals__ is globals_dict and value.__code__ == child
                    and value.__code__.co_filename == child.co_filename
                    and value.__code__.co_linetable == child.co_linetable,
                    'handoff_fixed_function:' + child.co_qualname + ':' + repr({
                        'globals': value.__globals__ is globals_dict, 'code': value.__code__ == child,
                        'filename': (value.__code__.co_filename, child.co_filename),
                        'flags': (value.__code__.co_flags, child.co_flags)}))
        else:
            require(child.co_name.startswith('<'), 'handoff_fixed_member_missing')


def parts(c: Any, v: Any, h: Any, t: Any, o: Any) -> Any:
    values = dict(C=c, V=v, H=h, T=t, O=o)
    for name, value in values.items():
        relative, expected = FIXED[name]
        path = ROOT / relative
        require(type(value) is ModuleType and Path(value.__file__).resolve() == path.resolve(), 'handoff_module_origin')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, 'handoff_source_changed')
        code_members(value, compile(data, str(path), 'exec', dont_inherit=True), vars(value))
    require(c.T is t and c.H is h and v.T is t and v.H is h, 'handoff_parts_identity')
    return SimpleNamespace(**values)
