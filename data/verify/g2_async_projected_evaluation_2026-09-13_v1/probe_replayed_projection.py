"""元全prefix再計算中の実Laneを投影へ接続。終端・live認証は昇格しない。"""
from copy import copy, deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import probe_actual_prefix_replay as OLD

ROOT = OLD.ROOT
OUT = ROOT / 'replayed_projection_v1'
TARGET_FRAME = 35672


def install(stack: Any, parts: Any, captured: list) -> None:
    replay = sys.modules['_g2_prefix_replay']
    owner = sys.modules[OLD.T.A.A.A.A.V4.OWNED_ALIAS]
    load = owner.bootstrap().load
    view = load('_async_replay_projected_view', ROOT / 'projected_view.py', {'belief': parts.mode.B})
    projection = load('_async_replay_prefix_projection', ROOT / 'prefix_projection.py',
        dict(identified_origin_candidate=replay.A.O, lane_state=replay.N, projected_view=view))
    original = replay.Replay.origins
    def origins(self: Any, ledger: Any, step: dict) -> None:
        if step['frame_idx'] == TARGET_FRAME:
            origin = next(e['active_origin'] for e in step['events'] if e.get('active_origin'))
            before = deepcopy((ledger, self.lane.families, self.lane.fired, self.lane.current))
            result = projection.project(parts, ledger, self.lane, step, origin)
            assert (ledger, self.lane.families, self.lane.fired, self.lane.current) == before
            assert not result.source_producer_authorized and not result.accounting_connected
            wrong = copy(self.lane)
            wrong.families = (replace(self.lane.families[0], phase=replay.N.P.SETTLED),)
            try:
                projection.project(parts, ledger, wrong, step, origin)
            except ValueError as error:
                assert str(error).endswith('projection_single_prepop_required')
            else:
                raise AssertionError('settledをprepopへ偽装して通過')
            captured.append(asdict(result))
        original(self, ledger, step)
    stack.callback(setattr, replay.Replay, 'origins', original)
    replay.Replay.origins = origins


def original_module() -> Any:
    sys.path.insert(0, str(OLD.RUNTIME))
    import target_entry as T
    OLD.T = T
    support = ModuleType('_async_replayed_projection_support')
    support.ROOT, support.T, support.R = ROOT, T, T.A
    support.read, support.lines = OLD.read, OLD.lines
    previous = sys.modules.get('review_arrivals')
    try:
        sys.modules['review_arrivals'] = support
        spec = importlib.util.spec_from_file_location('_async_replayed_projection_original', OLD.ORIGINAL)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
    finally:
        if previous is None: sys.modules.pop('review_arrivals', None)
        else: sys.modules['review_arrivals'] = previous
    result.X.ROOT = OUT
    return result


def main() -> None:
    assert not OUT.exists()
    OUT.mkdir()
    original = original_module()
    base_replay, base_inputs = original.replay, original.inputs
    captured = []
    def replay(stack: Any) -> dict:
        def inputs(parts: Any) -> tuple:
            values = base_inputs(parts)
            install(stack, parts, captured)
            return values
        original.inputs = inputs
        try:
            return base_replay(stack)
        finally:
            original.inputs = base_inputs
    original.replay = replay
    original.main()
    assert len(captured) == 1
    paths = (Path(__file__), ROOT / 'projected_view.py', ROOT / 'prefix_projection.py')
    result = dict(projections=captured, whole_saved_prefix_replayed=True, candidate_replayed=True,
        source_frame=TARGET_FRAME, wrong_phase_rejected=True, current_and_ledger_unchanged=True,
        live_registry_authorized=False, physical_GT_verified=False, model_evaluated=False,
        quality_gate_clear=False, code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with (OUT / 'CONDITIONAL_PROJECTION.json').open('x') as stream:
        json.dump(result, stream)
    print(json.dumps(dict(frame=TARGET_FRAME, outcomes=len(captured[0]['view']['outcomes']),
        candidate_replayed=True, current_unchanged=True, quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
