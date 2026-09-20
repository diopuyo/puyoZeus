"""G3候補の観測窓を、元Sinkの生成前に一体で束縛する。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from scripts import g3_video38_entry as G

OLD_FIRST, OLD_LAST = 29052, 36900
LAST = G.END - G.STRIDE
FRAMES = tuple(range(G.FIRST, G.END, G.STRIDE))
WINDOW_SOURCE = G.ROOT / 'data/verify/g2_history_prediction_diagnosis_2026-09-13_v1/observer_window_candidate.py'


def rebind(stack: Any, window: Any, values: dict, replace: Callable) -> dict:
    """元bindの認証後、設定元と検査器の期待値を同じscopeで変更する。"""
    scope, context, metadata = (values[key] for key in ('scope', 'context', 'metadata'))
    expected = (((OLD_FIRST, OLD_LAST),), OLD_FIRST, OLD_LAST, OLD_FIRST, OLD_LAST)
    actual = (scope.WINDOWS, context.OUTER_FIRST, context.OUTER_LAST, metadata.FIRST, metadata.LAST)
    G.require(actual == expected, 'observer_original_owner_bounds')
    G.require((window.FIRST, window.LAST, window.STRIDE) == (OLD_FIRST, OLD_LAST, G.STRIDE)
              and tuple(window.FRAMES) == tuple(range(OLD_FIRST, OLD_LAST + G.STRIDE, G.STRIDE)),
              'observer_original_verifier_bounds')
    G.require(context.STRIDE == metadata.STRIDE == G.STRIDE and metadata.FPS == G.FPS,
              'observer_source_clock')
    changes = [(scope, 'WINDOWS', ((G.FIRST, LAST),)), (context, 'OUTER_FIRST', G.FIRST),
               (context, 'OUTER_LAST', LAST), (metadata, 'FIRST', G.FIRST), (metadata, 'LAST', LAST),
               (window, 'FIRST', G.FIRST), (window, 'LAST', LAST), (window, 'FRAMES', FRAMES)]
    receipt = dict(first=G.FIRST, last=LAST, updates=len(FRAMES), quality_gate_clear=False)
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in changes]
    def restored() -> None:
        receipt['inner_binding_restored'] = all(getattr(owner, name) is value for owner, name, value in previous)
    stack.callback(restored)
    for owner, name, value in changes:
        replace(stack, owner, name, value)
    return receipt


def install(stack: ExitStack, adapter: Any, main: Any) -> None:
    """元bind/NEXT設定を通した後、Sink生成前の実設定元へ接続する。"""
    window, replace = adapter.OC.ORIGINAL, adapter.A.A.A.V4.replace_owned
    G.require(Path(window.__file__).resolve() == WINDOW_SOURCE.resolve(), 'observer_window_owner')
    original, receipts, outputs = window.bind, [], []
    def save_end() -> None:
        if outputs:
            G.save(outputs[0] / 'G3_OBSERVER_SCOPE_END.json', receipts[0])
    stack.callback(save_end)
    def bind(owner_stack: Any, latest: Any, owner_replace: Callable) -> dict:
        G.require(owner_stack is stack and owner_replace is replace, 'observer_stack_owner')
        values = original(owner_stack, latest, owner_replace)
        receipts.append(rebind(owner_stack, window, values, owner_replace))
        return values
    replace(stack, window, 'bind', bind)
    whole = adapter.A.A.A.V4.A
    original_bridge = whole.W.Bridge
    def bridge(collector: Any, state: dict, *args: Any, **kwargs: Any) -> Any:
        G.require(len(receipts) == 1, 'observer_binding_count')
        verified = adapter.OC.verify_state(state)
        G.require(verified['first'] == G.FIRST and verified['last'] == LAST, 'observer_verified_bounds')
        G.require(main.__globals__['K'].bounds()['first_frame'] == G.FIRST, 'observer_collector_first')
        outputs.append(state['output'])
        G.save(state['output'] / 'G3_OBSERVER_SCOPE_START.json', verified)
        return original_bridge(collector, state, *args, **kwargs)
    replace(stack, whole.W, 'Bridge', bridge)


def main() -> int:
    """原entryの実guard/保存/閉鎖を再利用する候補専用入口。"""
    G.require(G.arguments().arm == 'candidate', 'observer_candidate_only')
    original = G.protect_runtime
    def protect(stack: ExitStack, adapter: Any, entry: Any, plan: dict) -> None:
        G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'observer_entry_pin_required')
        original(stack, adapter, entry, plan)
        install(stack, adapter, entry)
    G.protect_runtime = protect
    try:
        return G.main()
    finally:
        G.protect_runtime = original


if __name__ == '__main__':
    raise SystemExit(main())
