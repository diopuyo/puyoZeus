"""検収済みAdmission入口の外側へG3現在評価資格を追加する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import g3_admission_entry as E
from scripts import g3_current_exit as C


def main() -> int:
    """元入口/guard/終了処理を再用し、稼働する別processの版を変更しない。"""
    args = E.G.arguments()
    plan = E.G.read(args.plan)
    for module in (C,):
        E.G.require(str(Path(module.__file__).resolve()) in plan['entry_pins'], 'current_exit_pin')
    E.G.require(str(Path(__file__).resolve()) in plan['entry_pins'], 'current_exit_entry_pin')
    original_install = E.install
    def install(stack: Any, adapter: Any) -> None:
        original_install(stack, adapter)
        whole, replace = adapter.A.A.A.V4.A, adapter.A.A.A.V4.replace_owned
        original_bridge = whole.W.Bridge
        def bridge(collector: Any, state: dict, inner: Any, *a: Any, **kw: Any) -> Any:
            value = original_bridge(collector, state, inner, *a, **kw)
            C.install(inner, collector, state, replace)
            E.G.save(state['output'] / 'G3_CURRENT_EXIT_ENTRY.json', dict(
                installs=1, admission_frames=state[C.A.KEY].frames,
                order='original_bridge -> admission.install -> current_exit.install',
                original_result_unchanged=True, model_evaluation_connected=False,
                quality_gate_clear=False))
            return value
        replace(stack, whole.W, 'Bridge', bridge)
    E.install = install
    try:
        return E.main()
    finally:
        E.install = original_install


if __name__ == '__main__':
    raise SystemExit(main())
