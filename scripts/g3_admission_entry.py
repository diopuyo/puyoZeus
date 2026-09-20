"""検収済み観測窓と当該OCR採録資格をvideo38実入口へ接続する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import g3_admission as A
from scripts import g3_observer_scope as S

G = S.G


def install(stack: Any, adapter: Any) -> None:
    """CPU停止wrapperより前に装着し、採録資格は実collectorの内側scopeへ置く。"""
    whole, replace = adapter.A.A.A.V4.A, adapter.A.A.A.V4.replace_owned
    original = whole.W.Bridge
    def bridge(collector: Any, state: dict, inner: Any, *args: Any, **kwargs: Any) -> Any:
        value = original(collector, state, inner, *args, **kwargs)
        G.require(value.collector is collector and value.state is state and value.stack is inner,
                  'admission_actual_bridge_owner')
        observer = A.install(inner, collector, state, replace, source_id='sha256:' + G.SOURCE_SHA)
        G.save(state['output'] / 'G3_ADMISSION_ENTRY.json', dict(
            source_id=observer.source_id, run_id=observer.run_id,
            order='cpu_stop -> admission -> observer_scope -> original_bridge; return -> admission.install',
            actual_inner_stack=True, installs=1, quality_gate_clear=False))
        return value
    replace(stack, whole.W, 'Bridge', bridge)


def main() -> int:
    """元entryのguard/終了保存を再利用し、追加コードもplanで固定する。"""
    G.require(G.arguments().arm == 'candidate', 'admission_candidate_only')
    original = G.protect_runtime
    def protect(stack: Any, adapter: Any, entry: Any, plan: dict) -> None:
        for module in (A, S):
            G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'admission_dependency_pin')
        G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'admission_entry_pin')
        original(stack, adapter, entry, plan)
        S.install(stack, adapter, entry)
        install(stack, adapter)
    G.protect_runtime = protect
    try:
        return G.main()
    finally:
        G.protect_runtime = original


if __name__ == '__main__':
    raise SystemExit(main())
