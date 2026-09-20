"""同runの確率更新と元contextのM1一次入力を照合。評価許可は発行しない。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent/'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v45'
BELIEF = ROOT.parent/'g2_probabilistic_scope_candidate_2026-09-11_v1'
sys.path.insert(0, str(BELIEF))
import serialization as S


def load_binding() -> Any:
    path = ROOT.parent/'g2_provisional_context_capture_2026-09-09_v1/binding.py'
    spec = importlib.util.spec_from_file_location('_actual_saved_m1_binding', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (RUN/name).read_text().splitlines()]


def check(binding: Any, row: Any, value: Any) -> dict[str, Any]:
    registration = dict(source_id=value.scope[0], run_id=value.scope[1],
        time_base_numerator=1, time_base_denominator=60, ledger_connection='NOT_CONNECTED')
    binding._identity(row, registration)
    binding._update(row)
    binding._generation(row)
    inputs = binding._inputs(row)
    observed = row['sides']['1P']['before_hold']['confirmed']['grid']
    visible_match = all([list(r) for r in world.grid[S.B.HIDDEN_ROWS:]]
                        == observed[S.B.HIDDEN_ROWS:] for world in value.worlds)
    return dict(frame=row['frame_idx'], belief_frame=value.frame,
        visible_match=visible_match, worlds=len(value.worlds),
        board_shape=list(inputs.boards.shape), queue_shape=list(inputs.queues.shape),
        probability_second_side_present=row['sides']['2P']['before_hold']['probability'] is not None,
        old_holds={side: row['sides'][side]['hold_reasons'] for side in ('1P', '2P')},
        integrity_and_queue_input_pass=True, evaluation_eligible=False)


def main() -> None:
    binding = load_binding()
    tracking = [r for r in rows('PROBABILISTIC_TRACKING.jsonl')
                if r['physical_transition_applied']]
    value = S.decode(tracking[-1]['transition']['state'])
    selected = [r for r in rows('provisional_context.jsonl') if r['frame_idx'] == value.frame]
    assert len(selected) == 1
    result = check(binding, selected[0], value)
    result.update(actual_model_evaluated=False, quality_gate_clear=False,
                  live_registry_bound=False, source_run=str(RUN))
    with (ROOT/'SAVED_INPUTS_RESULT_v1.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
