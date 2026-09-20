"""元全prefix再計算中の実Laneを投影へ接続。終端・live認証は昇格しない。"""
from copy import copy, deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import probe_actual_prefix_replay as OLD
from projection_origin_selector import selected_origin

ROOT = OLD.ROOT
OUT = ROOT / 'replayed_projected_input_v4'
TARGET_FRAME = 35672


def install(stack: Any, parts: Any, captured: list) -> None:
    replay = sys.modules['_g2_prefix_replay']
    owner = sys.modules[OLD.T.A.A.A.A.V4.OWNED_ALIAS]
    load = owner.bootstrap().load
    view = load('_async_replay_projected_view', ROOT / 'projected_view.py', {'belief': parts.mode.B})
    projection = load('_async_replay_prefix_projection', ROOT / 'prefix_projection.py',
        dict(identified_origin_candidate=replay.A.O, lane_state=replay.N, projected_view=view))
    packet = load('_async_replay_projected_input', ROOT / 'prefix_projected_input.py')
    notice = load('_async_replay_settled_notice', ROOT.parent / 'g2_second_origin_postrun_2026-09-13_v1/settled_notice.py')
    original = replay.Replay.advance
    def advance(self: Any, ledger: Any, row: dict, step: dict, qualification: Any) -> tuple:
        returned = original(self, ledger, row, step, qualification)
        ledger = self.R.finished_row(returned[1], row, step)
        if step['frame_idx'] == TARGET_FRAME:
            origin = selected_origin(step, ledger.scope[-1], 'step:6620', 130710687145984)
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
            wrapped = packet_input(packet, notice, parts, replay, projection, self.lane, ledger, step)
            assert wrapped['status'] == 'READY' and wrapped['prediction'] == json.loads(json.dumps(asdict(result)))
            absent = deepcopy(step)
            absent['returned']['active_origin'] = None
            held = packet_input(packet, notice, parts, replay, projection, self.lane, ledger, absent)
            assert held['status'] == 'HOLD' and held['prediction'] is None
            captured.append(wrapped['prediction'])
        return returned
    stack.callback(setattr, replay.Replay, 'advance', original)
    replay.Replay.advance = advance


def packet_input(packet: Any, notice: Any, parts: Any, replay: Any, projection: Any,
                 lane: Any, ledger: Any, step: dict) -> dict:
    """所有は人工proxy。原replayで構築したLaneと原Jの選択/投影/DTOだけ検査する。"""
    key = 'probabilistic_tracking_mode'
    source = SimpleNamespace(MODE_KEY=key, read=lambda *args: (lane, step, None))
    session = SimpleNamespace(state={key: SimpleNamespace(arrival_ledger=ledger)})
    live = SimpleNamespace(parts=parts, module=parts.mode)
    lease = SimpleNamespace(current=lambda: (live, replay.A))
    value = packet.read(session, lease, None, source, projection, step['frame_idx'], settled=notice.is_settled)
    assert not value['quality_gate_clear'] and not value['future_fire_power_supply_authorized']
    return json.loads(json.dumps(value))


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
    paths = (Path(__file__), ROOT / 'projected_view.py', ROOT / 'prefix_projection.py',
        ROOT / 'projection_origin_selector.py', ROOT / 'prefix_projected_input.py')
    result = dict(projections=captured, whole_saved_prefix_replayed=True, candidate_replayed=True,
        target_call_frame=TARGET_FRAME, hook_position='after_original_advance_and_finished_clock', wrong_phase_rejected=True, current_and_ledger_unchanged=True,
        live_registry_authorized=False, physical_GT_verified=False, model_evaluated=False,
        quality_gate_clear=False, code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with (OUT / 'CONDITIONAL_PROJECTION.json').open('x') as stream:
        json.dump(result, stream)
    print(json.dumps(dict(frame=TARGET_FRAME, outcomes=len(captured[0]['view']['outcomes']),
        candidate_replayed=True, current_unchanged=True, quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
