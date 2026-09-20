"""実SM資格入口へ保存済み採録ゲートを追加する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from scripts import g3_live_current_entry as R
from scripts import g3_current_admission as A

G = R.G


def main() -> int:
    """live guard設置直後、初回frame前の同stackへ束縛する。"""
    plan = G.read(G.arguments().plan)
    for path in (Path(__file__).resolve(), Path(A.__file__).resolve()):
        G.require(str(path) in plan['entry_pins'], 'current_admission_entry_pin')
    original = R.L.install
    def install(stack: Any, state: dict, replace: Any) -> None:
        original(stack, state, replace)
        gate = A.install(stack, state, replace)
        G.save(Path(state['output']) / 'G3_CURRENT_ADMISSION_ENTRY.json', dict(
            current_rows=gate.current.rows, admission_decisions=gate.observer.decisions,
            original_emit_unchanged=True, quality_gate_clear=False))
    R.L.install = install
    try:
        return R.main()
    finally:
        R.L.install = original


if __name__ == '__main__':
    raise SystemExit(main())
