"""原cold入口の2更新だけでraw guard→Capture完了→最終保存/解除を検査する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any
import probe_imports as I
import collector_connector as C
import runtime_session as S

ROOT = Path(__file__).resolve().parent
STOP = 'joint_capture_after_two_original_updates'
UPDATES = 2


def main(proof_name: str = 'TWO_UPDATE_PROBE_v1.json') -> None:
    result: dict[str, Any] = dict(frames=[], actual_live_registry_evaluated=False, quality_gate_clear=False)
    references: dict[str, Any] = {}
    with ExitStack() as stack:
        C.install(stack)
        facade = C.INTEGRATED.OLD
        def instrument(state: dict, capture: Any) -> Any:
            owner = state['joint_capture_stack']
            loop, original = capture.loop, capture.loop.collect_lean
            references.update(capture=capture, loop=loop, original=capture.original, state=state)
            def restore() -> None:
                assert loop.collect_lean is collect, 'probe_foreign_collect'
                loop.collect_lean = original
            def collect(*args: Any, **kwargs: Any) -> Any:
                value = original(*args, **kwargs)
                snapshot = capture.snapshot()
                result['frames'].append(snapshot['last_frame'])
                result['last_observed_count'] = snapshot['observed_count']
                result['metadata_count'] = capture.tail.count
                if snapshot['observed_count'] == UPDATES:
                    raise RuntimeError(STOP)
                return value
            owner.callback(restore)
            loop.collect_lean = collect
            return C.final_capture(state, capture)
        facade.final_capture = instrument
        I.R.V6.V4.start = S.start
        I.R.V6.V5.create = S.create
        code = I.R.V6.V5.main()
    if references:
        result.update(capture_closed=references['capture'].closed,
            collector_restored=references['loop'].collect_lean is references['original'],
            generator_closed=references['loop'].generator.gi_frame is None,
            capture_save_error=references['state'].get('joint_capture_save_error'),
            saved_capture=str(references['state']['output'] / 'JOINT_PRODUCER_CAPTURE.json'))
    result.update(original_exit=code, planned_stop=STOP)
    with (ROOT / proof_name).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    assert code == 1 and len(result['frames']) == UPDATES
    assert result['capture_closed'] and result['collector_restored'] and result['generator_closed']
    assert result['capture_save_error'] is None and Path(result['saved_capture']).is_file()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
