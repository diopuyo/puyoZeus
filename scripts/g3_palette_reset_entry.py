"""現在/採録gateを設置した実入口へpalette reset保留だけを追加する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from scripts import g3_current_admission_entry as R
from scripts import g3_palette_reset_hold as P

G = R.G


def main() -> int:
    """palette実Recorderのある同stackへ束縛し、元入口をfinallyで復元する。"""
    plan = G.read(G.arguments().plan)
    for path in (Path(__file__).resolve(), Path(P.__file__).resolve()):
        G.require(str(path) in plan['entry_pins'], 'palette_reset_entry_pin')
    original = R.A.install
    def install(stack: Any, state: dict, replace: Any) -> Any:
        gate = original(stack, state, replace)
        hold = P.install(stack, state, replace)
        G.save(Path(state['output']) / 'G3_PALETTE_RESET_ENTRY.json', dict(
            module=hold.module.__name__, source_sha=P.SOURCE_SHA, original_wrapper_unchanged=True,
            current_rows=gate.current.rows, admission_decisions=gate.observer.decisions,
            quality_gate_clear=False))
        return gate
    R.A.install = install
    try:
        return R.main()
    finally:
        R.A.install = original


if __name__ == '__main__':
    raise SystemExit(main())
