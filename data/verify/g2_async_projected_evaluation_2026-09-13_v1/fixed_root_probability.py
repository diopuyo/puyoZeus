"""同runの確率1票で固定初期rootを裏付け、既存の全消化後予測だけを再用する。"""
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any
import fixed_origin_reference as F

SIDE, FPS, ROWS, COLS = '2P', 60, 13, 6
COLORS = frozenset((0, 1, 2, 3, 4, 5, 9))
OWNER_KEYS = ('source_id', 'run_id', 'pipe_object_id')


def require(value: bool, reason: str) -> None:
    if not value: raise ValueError('fixed_root_probability:' + reason)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def packet(frame: int, reason: str) -> dict:
    return dict(kind='conditional_fixed_root_input/v1', side=SIDE, cutoff_frame=frame,
        status='HOLD', reason=reason, meaning='fixed_root_fully_resolved_not_remaining_attack',
        accounting_permission=False, future_fire_power_supply_authorized=False,
        physical_identity_verified=False, calibrated_win_probability=False, quality_gate_clear=False,
        native_current_changed=False, additional_simulation=False, prediction=None)


def owner(history: Any, capture: Any, frame: int) -> None:
    require(not history.closed and history.error is None and history.sealed and history.transferred,
            'history_lifetime')
    require(history.probability is not None and history.probability.closed, 'early_capture_not_sealed')
    require(history.journal is capture.journal and history.pipe is capture.pipe, 'history_owner')
    require(history.last_frame <= frame, 'history_future')


def certain_grid(probability: dict) -> tuple | None:
    """13x6・row0を含む整数色をそのまま使う。空=0、おじゃま=9、UNKNOWNは不採用。"""
    require(probability['present'] and probability['type_valid'] and not probability['errors'], 'invalid_saved_PB')
    cells = probability['cells']
    require(isinstance(cells, list) and len(cells) == ROWS
            and all(isinstance(row, list) and len(row) == COLS for row in cells), 'PB_shape')
    require(digest(cells) == probability['sha256'], 'PB_hash')
    if not all(isinstance(cell, list) and len(cell) == 1 and len(cell[0]) == 2
               and type(cell[0][0]) is int and cell[0][0] in COLORS
               and type(cell[0][1]) in (int, float) and cell[0][1] == 1.0 for row in cells for cell in row):
        return None
    return tuple(tuple(cell[0][0] for cell in row) for row in cells)


def qualification(value: dict, capture: Any, snapshot: Any, history: Any, frame: int) -> str | None:
    scope = capture.scopes[SIDE]
    require(all(value['journal_scope'][key] == scope[key] for key in OWNER_KEYS), 'PB_source_owner')
    expected = [scope['source_id'], scope['run_id'], scope['software_reset'], id(capture.pipe),
                id(capture.pipe._sm_2p), snapshot.generation.reset_epoch, SIDE]
    generation = asdict(snapshot.generation)
    if (value['scope'] != expected or {key: value['generation'][key] for key in generation} != generation
            or value['generation_after'] != value['generation']): return 'probability_generation_mismatch'
    require(value['side'] == value['journal_scope']['side'] == SIDE, 'PB_side')
    require(type(value['frame']) is int
            and value['clock'] == value['journal_scope']['time_sec'] == value['frame'] / FPS, 'PB_clock')
    require(value['frame'] == value['journal_scope']['frame_idx'] <= history.last_frame <= frame, 'PB_future')
    if value['frame'] < math.ceil(snapshot.episodes[0].trigger_sec * FPS): return 'probability_before_origin_trigger'
    require(value['state'] == 'stable' and not value['pb_holds'] and not value['pb_errors']
            and not value['in_step_generation_changed'], 'unqualified_archived_candidate')
    return None


def read(history: Any, capture: Any, frame: int) -> dict:
    owner(history, capture, frame)
    reference = F.read(capture, SIDE, frame)
    if reference['status'] != 'REFERENCE_ONLY': return packet(frame, reference['reason'])
    snapshot = capture.ledger.snapshot(capture.handles[SIDE])
    origin = reference['origin']
    if origin['available_at']['frame_idx'] != history.last_frame:
        return packet(frame, 'origin_not_from_early_history')
    require(origin['input_sha256'] == snapshot.origin_before_sha256 == digest(origin['input_grid']), 'root_hash')
    require(tuple(map(tuple, origin['input_grid'])) == snapshot.origin_before_grid, 'root_grid')
    require(origin['final_sha256'] == digest(origin['final_grid']), 'prediction_final_hash')
    reasons, matching = set(), []
    for raw in history.probability_inputs():
        value = json.loads(raw)
        reason = qualification(value, capture, snapshot, history, frame)
        if reason is None:
            grid = certain_grid(value['probability'])
            reason = 'joint_not_certified' if grid is None else 'probability_root_grid_mismatch' if grid != snapshot.origin_before_grid else None
        if reason is not None: reasons.add(reason)
        else: matching.append(value)
    if not matching: return packet(frame, ';'.join(sorted(reasons)) or 'qualified_early_probability_missing')
    selected = min(matching, key=lambda value: (value['frame'], value['journal_token']))
    result = packet(frame, 'conditional_on_exact_certified_initial_root')
    result.update(status='READY', probability_frame=selected['frame'], history_last_frame=history.last_frame,
        origin_trigger_sec=snapshot.episodes[0].trigger_sec, instance_id=reference['instance_id'],
        probability_source=selected, probability_source_digest=digest(selected),
        origin_input_sha256=origin['input_sha256'], origin_prediction_revision=origin['prediction_revision'],
        prediction=dict(grid=origin['final_grid'], final_sha256=origin['final_sha256'],
            chain_count=origin['chain_count'], raw_chain_score=origin['calculated_total_score'], weight=1.0))
    return result
