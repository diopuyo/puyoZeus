"""既存reset入口へ実SM現在資格を最小接続する。GPU再走許可は発行しない。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from scripts import g3_reset_entry as R
from scripts import g3_live_current_guard as L

G = R.G


def main() -> int:
    """実Bridgeの同所有stackで出口設置直後に防御を追加する。"""
    plan = G.read(G.arguments().plan)
    for path in (Path(__file__).resolve(), Path(L.__file__).resolve()):
        G.require(str(path) in plan['entry_pins'], 'live_current_entry_pin')
    original = L.C.install
    def install(stack: Any, collector: Any, state: dict, replace: Any) -> Any:
        result = original(stack, collector, state, replace)
        L.install(stack, state, replace)
        G.save(Path(state['output']) / 'G3_LIVE_CURRENT_ENTRY.json', dict(
            installed_before_frames=result.rows == 0, original_result_unchanged=True,
            model_evaluation_connected=False, quality_gate_clear=False))
        return result
    L.C.install = install
    try:
        return R.main()
    finally:
        L.C.install = original


if __name__ == '__main__':
    raise SystemExit(main())
