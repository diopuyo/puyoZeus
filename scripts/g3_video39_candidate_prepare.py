"""source39の実候補入口をCPU準備だけで検証する。GPU開始は明示的に拒否。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

from scripts import g3_postcommit_entry as E
from scripts import g3_video39_candidate_scope as S

G, B = E.G, S.B
WHOLE_SOURCE = G.ROOT / 'data/verify/g2_live_probability_context_2026-09-12_v1/whole_collector_capture.py'
WHOLE_SHA = 'fed683df0925cfb8f24896adf075ab224ac12d659e718dc812ee3e7df203eb1b'
WRAP_LIMIT = 8


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    stack.callback(setattr, owner, name, getattr(owner, name))
    setattr(owner, name, value)


def identify(value: Any) -> str:
    """差し替え済みBridgeの出所を失敗票へ残す。欠測を成功へ変換しない。"""
    code = getattr(value, '__code__', None)
    cells = [type(cell.cell_contents).__name__ for cell in (getattr(value, '__closure__', None) or ())]
    return '|'.join(str(item) for item in (type(value).__name__, getattr(value, '__qualname__', None),
        getattr(value, '__module__', None), code and code.co_filename, code and code.co_firstlineno, cells))


def original_bridge(module: Any) -> type:
    """実runtimeが正規に包んだBridgeから、元moduleで定義された元classだけを解決する。"""
    value, depth = module.Bridge, 0
    while not isinstance(value, type):
        found = [cell.cell_contents for cell in (getattr(value, '__closure__', None) or ())
                 if isinstance(cell.cell_contents, type)
                 and cell.cell_contents.__module__ == module.__name__
                 and cell.cell_contents.__qualname__ == 'Bridge']
        G.require(len(found) == 1, 'source39_whole_bridge_owner:' + identify(value))
        value, depth = found[0], depth + 1
        G.require(depth <= WRAP_LIMIT, 'source39_whole_bridge_depth')
    G.require(value.__module__ == module.__name__ and value.__qualname__ == 'Bridge',
              'source39_whole_bridge_identity')
    return value


def whole_clock(stack: Any, adapter: Any) -> None:
    """元Bridgeの同じglobalsだけを設定し、frame/二つのclock検査を残す。"""
    module = adapter.A.A.A.V4.A.W
    G.require(Path(module.__file__).resolve() == WHOLE_SOURCE and B.M.digest(WHOLE_SOURCE) == WHOLE_SHA,
              'source39_whole_source')
    G.require(sys.modules.get(module.__name__) is module, 'source39_whole_namespace')
    G.require((module.FPS, module.STRIDE) == (S.OLD_FPS, S.OLD_STRIDE), 'source39_whole_old_clock')
    for name in ('__init__', 'observe', 'observe_physical'):
        G.require(getattr(original_bridge(module), name).__globals__ is vars(module),
                  'source39_whole_function_globals')
    replace(stack, module, 'FPS', B.FPS)
    replace(stack, module, 'STRIDE', B.STRIDE)


def install(stack: ExitStack, output: Path | None = None) -> None:
    """別process内で元source39コンテナ検査と候補収集設定を組み合わせる。"""
    source = B.bind(stack)
    replace(stack, G, 'bind_bounds', S.bind_bounds)
    replace(stack, S.O, 'rebind', S.observer_rebind)
    original_capture = G.capture_scope
    function = B.checked_capture.__wrapped__
    namespace = dict(function.__globals__, G=N(**(vars(G) | dict(capture_scope=original_capture))))
    checked = contextmanager(FunctionType(function.__code__, namespace, function.__name__,
                                         function.__defaults__, function.__closure__))
    def capture(collector: Any, state: dict) -> Any:
        return checked(collector, source, state)
    replace(stack, G, 'capture_scope', capture)
    def receipt(plan: dict, output: Path, base: Any) -> tuple[dict, dict]:
        G.require(B.M.digest(Path(base.__file__)) == B.BASE_SHA, 'source39_base')
        replace(stack, base, 'collection_arguments', S.arguments)
        return S.receipt_for(plan, output, base)
    replace(stack, G, 'receipt_for', receipt)
    previous, identity = G.protect_runtime, {}
    def protect(owner_stack: Any, adapter: Any, main: Any, fixed: dict) -> None:
        # 元Bridgeのglobals検査は前段層がBridgeを包む前に行う。
        whole_clock(owner_stack, adapter)
        identity.update(S.install_source_identity(owner_stack, adapter))
        def save_identity() -> None:
            if output is not None:
                G.save(output / 'G3_SOURCE39_IDENTITY.json',
                       dict(identity, replacements=len(identity.get('receipts', []))))
        owner_stack.callback(save_identity)
        previous(owner_stack, adapter, main, fixed)
    replace(stack, G, 'protect_runtime', protect)


def main() -> int:
    """この段階の成功をsource39 driver/finalizerの完成へ読み替えない。"""
    args = G.arguments()
    G.require(args.cpu_prepare and args.arm == 'candidate', 'source39_cpu_prepare_only')
    plan = G.read(args.plan)
    for module in (sys.modules[__name__], S, B):
        G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'source39_prepare_pin')
    with ExitStack() as stack:
        install(stack, Path(args.output))
        return E.main()


if __name__ == '__main__':
    raise SystemExit(main())
