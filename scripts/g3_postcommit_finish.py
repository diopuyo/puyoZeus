"""G3の実終了readerとworld索引だけを、元判定を残して接続する。"""
from __future__ import annotations

from pathlib import Path
import sys
from types import CodeType
from typing import Any

from scripts import g3_postcommit_binding as B
from scripts import g3_postcommit_rows as R
from scripts.g3_context_rows import digest, require

ROOT = Path(__file__).resolve().parents[1] / 'data/verify'
PINS = {
    'g2_conditional_finalizer_fusion_2026-09-10_v1/live_connection.py': 'c7eae46f1b37daef6ba3400bb4bb6d53e4d6edda881ea8a25c24e644e5ad4cd5',
    'g2_conditional_finalizer_fusion_2026-09-10_v1/fusion.py': '32e17883146f3922be45b170d66be18404ea4b7b1768971ca56837fb5e3b983e',
    'g2_conditional_finalizer_world_2026-09-10_v1/world_api.py': 'a99e78c9fa3e0fb5dbcdddd122b5d519c7f4409b05c13ced515038a26d450858',
    'g2_live_probability_context_2026-09-12_v1/probability_boundary.py': 'e391e867416c003388d095c53bff8259281257f160dcd4c9c4b87399f44f91ef',
    'g2_private_suffix_live_adapter_2026-09-10_v1/live_finalizer.py': '876f0bf13155121b1774f4aea74bc759ec388a73fef9994dc0fe74464197dbb8',
    'g2_history_publication_probe_runtime_2026-09-10_v11/goals.py': '5c43fb5323cbfce1761a1ea84511f68b1b71b009208a5b6027253588480088d7',
}
MARKER = "json.loads((output / 'POSTCOMMIT_CONSUMER_ROWS.json').read_text())"
HELPER = '_G3_POSTCOMMIT_READ'


def checked(module: Any, relative: str) -> Any:
    """別aliasの同名コードを暗黙に採用しない。"""
    path = ROOT / relative
    require(Path(module.__file__).resolve() == path and digest(path) == PINS[relative],
            'postcommit_finish_source:' + relative)
    return module


def fallback(stack: Any, module: Any, rows: R.SavedRows, replace: Any) -> dict:
    """既存lambda等が捕捉する同一functionの、一つの保存read式だけを変更する。"""
    path = Path(module.__file__)
    raw = path.read_text(encoding='utf-8')
    original = module.evaluate
    expected = next(v for v in compile(raw, str(path), 'exec', dont_inherit=True).co_consts
                    if isinstance(v, CodeType) and v.co_name == 'evaluate')
    require(original.__code__ == expected and original.__globals__ is vars(module),
            'postcommit_fallback_function')
    require(raw.count(MARKER) == 1 and HELPER not in vars(module), 'postcommit_fallback_marker')
    generated = raw.replace(MARKER, HELPER + "(output / 'POSTCOMMIT_CONSUMER_ROWS.json')")
    code = next(v for v in compile(generated, str(path), 'exec', dont_inherit=True).co_consts
                if isinstance(v, CodeType) and v.co_name == 'evaluate')
    require(code.co_freevars == original.__code__.co_freevars, 'postcommit_fallback_closure')
    vars(module)[HELPER] = rows.saved
    stack.callback(vars(module).pop, HELPER)
    replace(stack, original, '__code__', code)
    return dict(source_sha256=digest(path), generated_source=generated,
                function_identity_preserved=module.evaluate is original)


def bind_index(stack: Any, module: Any, replace: Any, *, field: bool) -> None:
    original = module.indexed
    def indexed(rows: Any, key: Any) -> Any:
        return R.indexed(original, rows, key, field=field)
    replace(stack, module, 'indexed', indexed)


def world_index(stack: Any, fusion: Any, replace: Any) -> Any:
    """既存import scopeを即時閉鎖し、後段も再利用する実worldを選ぶ。"""
    before, paths = dict(sys.modules), list(sys.path)
    with fusion.modules() as (_, world):
        api = checked(world.A, 'g2_conditional_finalizer_world_2026-09-10_v1/world_api.py')
    require(sys.path == paths and all(sys.modules.get(k) is v for k, v in before.items()),
            'postcommit_world_import_noninterference')
    added = {k: v for k, v in sys.modules.items() if k not in before}
    def restore() -> None:
        for name, value in added.items():
            # 元session.configuredが先にproject内の新規aliasを解放する。
            path = getattr(value, '__file__', None)
            if name not in sys.modules and path and Path(path).resolve().is_relative_to(ROOT.parents[1]):
                continue
            require(sys.modules.get(name) is value, 'postcommit_world_alias_changed:' + name)
            sys.modules.pop(name)
    stack.callback(restore)
    bind_index(stack, api, replace, field=False)
    return world


def install(stack: Any, rows: R.SavedRows, replace: Any) -> dict:
    """実Bridge時点で確認済みの6接続を同じ外側所有scopeに置く。"""
    module = checked(sys.modules['_conditional_live_finalizer'],
                     'g2_conditional_finalizer_fusion_2026-09-10_v1/live_connection.py')
    fusion = checked(module.F, 'g2_conditional_finalizer_fusion_2026-09-10_v1/fusion.py')
    private = checked(sys.modules['_private_live_finalizer'],
                      'g2_private_suffix_live_adapter_2026-09-10_v1/live_finalizer.py')
    goals = checked(sys.modules['goals'], 'g2_history_publication_probe_runtime_2026-09-10_v11/goals.py')
    boundary = checked(sys.modules['probability_boundary'],
                       'g2_live_probability_context_2026-09-12_v1/probability_boundary.py')
    replace(stack, private, 'read', B.reader(private.read, rows))
    replace(stack, goals.K, 'read', B.reader(goals.K.read, rows))
    bind_index(stack, boundary, replace, field=True)
    world = world_index(stack, fusion, replace)
    generated = fallback(stack, module, rows, replace)
    return dict(fallback=generated, world_api=str(world.A.__file__),
                goals=str(goals.__file__), private_reader=str(private.__file__),
                boundary=str(boundary.__file__), quality_gate_clear=False)
