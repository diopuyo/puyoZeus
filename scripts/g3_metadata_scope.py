"""G3候補だけのmetadata区間接続。原producerと保存検査は再利用する。"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
from typing import Any

from scripts import g3_video38_entry as G

OBSERVER = G.ROOT / 'data/verify/g2_collector_metadata_capture_2026-09-09_v1/observer.py'
BOUNDED = G.ROOT / 'data/verify/g2_collector_metadata_bounded_2026-09-09_v1/bounded.py'
RUNTIME_BOUNDS = (29052, 36900, 2, 60)
G3_BOUNDS = (G.FIRST, G.END - G.STRIDE, G.STRIDE, G.FPS)
NAMES = ('FIRST', 'LAST', 'STRIDE', 'FPS')


def bind_metadata(stack: ExitStack, state: dict[str, Any],
                  expected: tuple[int, int, int, int] = RUNTIME_BOUNDS) -> dict:
    """実Sinkとクラスappendが使う同じ辞書を、外側scopeの寿命で束縛する。"""
    sink = state['collector_metadata_sink']
    rows = sink.rows
    namespace = sink.wrapper.__func__.__globals__
    append_globals = type(rows).append.__globals__
    G.require(Path(namespace['__file__']).resolve() == OBSERVER.resolve(), 'metadata_owner')
    G.require(Path(append_globals['__file__']).resolve() == BOUNDED.resolve(), 'metadata_rows_owner')
    G.require(vars(append_globals['O']) is namespace, 'metadata_shared_namespace')
    G.require(type(sink) is append_globals['Sink'] and type(rows) is append_globals['RowStream'],
              'metadata_actual_types')
    G.require(rows.count == 0 and not sink.closed and not sink.busy and not sink.errors,
              'metadata_fresh_sink')
    old = tuple(namespace[name] for name in NAMES)
    G.require(old == expected and namespace['SIDES'] == ('1P', '2P'), 'metadata_original_bounds')
    receipt = dict(before=list(old), applied=list(G3_BOUNDS), owner=str(OBSERVER),
                   same_namespace=True, restored=False, quality_gate_clear=False)
    def restore(kind: Any, error: Any, trace: Any) -> bool:
        current = tuple(namespace[name] for name in NAMES)
        namespace.update(zip(NAMES, old))
        receipt.update(restored=True, final_rows=rows.count, sink_closed=sink.closed,
                       changed_during_scope=current != G3_BOUNDS,
                       planned_stop=error is not None and str(error) == G.CPU_STOP,
                       error=None if error is None else dict(type=type(error).__name__, message=str(error)))
        G.save(state['output'] / 'G3_METADATA_SCOPE.json', receipt)
        G.require(current == G3_BOUNDS or error is not None, 'metadata_scope_interference')
        return False
    stack.push(restore)
    namespace.update(zip(NAMES, G3_BOUNDS))
    return receipt


def install_bridge(stack: ExitStack, adapter: Any) -> None:
    """内側collectのstateを実Bridge引数から受け取る。"""
    whole = adapter.A.A.A.V4.A
    original = whole.W.Bridge
    def bridge(collector: Any, state: dict, *args: Any, **kwargs: Any) -> Any:
        bind_metadata(stack, state)
        return original(collector, state, *args, **kwargs)
    whole.W.Bridge = bridge
    stack.callback(setattr, whole.W, 'Bridge', original)


def main() -> int:
    """原入口のguard/閉鎖を保持し、候補armだけへ追加接続する。"""
    args = G.arguments()
    G.require(args.arm == 'candidate', 'metadata_candidate_only')
    original = G.protect_runtime
    def protect(stack: ExitStack, adapter: Any, entry: Any, plan: dict) -> None:
        path = str(Path(__file__).resolve())
        G.require(plan['entry_pins'].get(path) == hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  'metadata_entry_pin')
        original(stack, adapter, entry, plan)
        install_bridge(stack, adapter)
    G.protect_runtime = protect
    try:
        return G.main()
    finally:
        G.protect_runtime = original


if __name__ == '__main__':
    raise SystemExit(main())
