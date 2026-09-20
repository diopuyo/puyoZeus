"""原入口で新connect/raw upgrade/Captureの初期化と解除を採録前に検査する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any
import probe_imports as I
import collector_connector as C
import runtime_session as S

ROOT = Path(__file__).resolve().parent
STOP = 'joint_connected_before_first_collect'


def main() -> None:
    result: dict[str, Any] = {}
    with ExitStack() as stack:
        C.install(stack)
        facade = C.INTEGRATED.OLD
        def observed(state: dict, capture: Any) -> Any:
            result.update(source_id=capture.identity['source_id'], run_id=capture.identity['run_id'],
                observed_count=capture.count, metadata_count=capture.tail.count,
                observer_type=type(capture.recorder).__name__,
                metadata_range=[capture.tail.original.FIRST, capture.tail.original.LAST],
                boundary_stack_identity_checked=False,
                actual_updates=0, quality_gate_clear=False)
            # close callback自体は原経路へ登録済み。ここは採録開始前の計画停止。
            raise RuntimeError(STOP)
        facade.final_capture = observed
        I.R.V6.V4.start = S.start
        I.R.V6.V5.create = S.create
        code = I.R.V6.V5.main()
    result.update(original_exit=code, connector_restored=C.INTEGRATED.OLD is not facade)
    with (ROOT / 'CONNECTED_CONSTRUCTOR_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    assert result.get('observed_count') == 0 and result.get('metadata_range') == [C.FIRST, C.LAST] and code == 1
    print(json.dumps(result))


if __name__ == '__main__':
    main()
