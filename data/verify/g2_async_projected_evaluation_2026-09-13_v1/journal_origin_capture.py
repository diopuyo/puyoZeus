"""既存Witnessの原票を別所有の既存予測台帳へ接続する診断consumer。"""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
from typing import Any, Callable
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
from src.chain_detector import CHAIN_MECHANISM_LANDING
import journal_pair_reader as R

SCOPE_KEYS = ('source_id', 'run_id', 'pipe_object_id', 'software_reset')
FPS = 60


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


class OriginCapture:
    def __init__(self, witness: Any, journal: Any, pipe: Any, ledger_module: Any,
                 simulator: Any, board_factory: Callable, settled: Callable, stream: Any, *,
                 projected_input: Callable[[int], dict] | None = None) -> None:
        self.witness, self.journal, self.pipe = witness, journal, pipe
        self.module, self.ledger = ledger_module, ledger_module.ChainPredictionLedger()
        self.simulator, self.board_factory, self.settled, self.stream = simulator, board_factory, settled, stream
        self.handles: dict[str, Any] = {}
        self.seen: dict[str, set[str]] = {side: set() for side in R.SIDES}
        self.origin_generations: dict[tuple, Any] = {}
        self.scopes: dict[str, dict] = {}
        self.scope_holds: dict[str, Any] = {}
        self.last_frame, self.closed, self.error, self.save_error = -1, False, None, None
        self.projected_input = projected_input

    def completed(self, frame: int) -> dict:
        if self.closed or self.error is not None or frame <= self.last_frame:
            raise ValueError('origin_capture_lifetime_or_duplicate')
        source_json = None
        try:
            source = R.read_pair(self.witness, self.journal, self.pipe, frame)
            source_json = source.rows_json
            rows = json.loads(source.rows_json)
            events = [self.consume(row) for row in rows]
            result = dict(frame=frame, holds=source.holds, sides=events, diagnostic_only=True,
                accounting_permission=False, quality_gate_clear=False, physical_identity_verified=False,
                future_fire_power_supply_authorized=False, original_pipeline_simulate=False,
                source_producer_authorized=False, live_hook_verified=False)
            if self.projected_input is not None:
                result['projected_input'] = self.projected_input(frame)
            self.stream.write(encoded(result) + '\n')
            self.stream.flush()
            self.last_frame = frame
            return result
        except BaseException as error:
            self.error = error
            self.record_failure(frame, source_json, error)
            raise

    def record_failure(self, frame: int, source: str | None, error: BaseException) -> None:
        try:
            self.stream.write(encoded(dict(kind='origin_capture_failure', frame=frame,
                error=repr(error), source_rows_json=source,
                partial_instances={side: handle.instance_id for side, handle in self.handles.items()},
                quality_gate_clear=False)) + '\n')
            self.stream.flush()
        except BaseException as save_error:
            self.save_error = save_error  # 元例外を置換しない。

    def consume(self, row: dict, *, available_frame: int | None = None) -> dict:
        side, generation = row['side'], row['generation_after']
        current = self.module.ChainGeneration(side, generation['reset_epoch'], generation['action_revision'])
        point = ledger_point(row, available_frame)
        scope = {key: row[key] for key in SCOPE_KEYS}
        old_scope = self.scopes.get(side)
        scope_changed = old_scope is not None and old_scope != scope
        if old_scope is not None:
            if any(old_scope[key] != scope[key] for key in SCOPE_KEYS[:-1]):
                raise ValueError('origin_capture_source_owner_changed')
            if scope['software_reset'] < old_scope['software_reset']:
                raise ValueError('origin_capture_scope_rewind')
        handle = self.handles.get(side)
        if handle is not None and (scope_changed or self.ledger.snapshot(handle).generation != current):
            previous = self.ledger.snapshot(handle).generation
            self.ledger.invalidate(handle, generation=previous,
                **point, reason=('software_scope_changed_not_physical_identity' if scope_changed
                                 else 'software_generation_changed_not_physical_identity'))
            del self.handles[side]
            self.seen[side].clear()
            if scope_changed: self.scope_holds[side] = previous
        self.scopes[side] = scope.copy()  # 返却DTOの変更を内部所有情報へ伝播させない。
        self.remember_origins(row, current if row['generation'] == generation else None)
        if self.scope_holds.get(side) == current:
            return dict(side=side, source_scope=scope, origin=None, instance_id=None,
                        reasons=['scope_changed_generation_not_advanced'])
        self.scope_holds.pop(side, None)
        if row['generation'] != generation:
            return dict(side=side, source_scope=scope, reason='generation_changed_within_step', origin=None)
        reasons = []
        for item in row['events']:
            raw = item.get('active_origin')
            if raw is not None and not self.settled(raw):
                reason = self.observe(side, current, point, raw,
                                      recorded_frame=None if available_frame is None else row['frame_idx'])
                if reason is not None: reasons.append(reason)
        handle = self.handles.get(side)
        origin = None if handle is None else _origin_prediction(self.ledger.snapshot(handle))
        return dict(side=side, source_scope=scope, reasons=sorted(set(reasons)),
            instance_id=None if handle is None else handle.instance_id,
            origin=None if origin is None else asdict(origin))

    def remember_origins(self, row: dict, generation: Any) -> None:
        """保留中も初出時点を保持する。これは台帳登録/物理所有の許可ではない。"""
        side = row['side']
        ownership = (generation, encoded(self.scopes[side]))
        for item in row['events']:
            raw = item.get('active_origin')
            if raw is not None and not self.settled(raw):
                identity = (side, raw['object_id'], raw['trigger_sec'])
                self.origin_generations.setdefault(identity, ownership)

    def observe(self, side: str, generation: Any, point: dict, raw: dict,
                *, recorded_frame: int | None = None) -> str | None:
        identity = (side, raw['object_id'], raw['trigger_sec'])
        previous = self.origin_generations.get(identity)
        ownership = (generation, encoded(self.scopes[side]))
        if previous is not None and previous != ownership:
            if previous[0] is None: return 'origin_observed_during_ambiguous_generation'
            return ('origin_from_retired_software_scope' if previous[1] != ownership[1]
                    else 'origin_from_retired_software_generation')
        key = encoded(raw)
        if key in self.seen[side]: return None
        saved = raw['before_board']
        if saved is None or hashlib.sha256(encoded(saved['grid']).encode()).hexdigest() != saved['sha256']:
            raise ValueError('origin_capture_source_board_hash')
        names = ('trigger_sec', 'end_sec', 'chain_count', 'total_score', 'mechanism', 'score_estimated')
        event = SimpleNamespace(**{name: raw[name] for name in names}, before_board=self.board_factory(saved))
        handle = self.handles.get(side)
        provenance = 'live_J_readonly_reprojection_not_original_simulate'
        if recorded_frame is not None:
            provenance = f'same_run_history_recorded_{recorded_frame}_received_{point["frame_idx"]}'
        if handle is None:
            if event.mechanism != CHAIN_MECHANISM_LANDING: return 'first_event_not_landing'
            handle = self.ledger.open_landing_provisional(generation=generation, **point,
                origin_before_board=event.before_board, landing_event=event, capture_source=provenance)
            self.handles[side] = handle
            episode = self.ledger.snapshot(handle).episodes[0]
        else:
            episode = self.ledger.add_episode(handle, generation=generation, **point,
                event=event, capture_source=provenance)
        prediction = self.simulator.simulate(event.before_board)
        self.ledger.add_prediction(handle, generation=generation, **point,
            episode_revision=episode.episode_revision, input_board=event.before_board, result=prediction)
        self.seen[side].add(key)
        self.origin_generations[identity] = ownership
        return None

    def close(self) -> None:
        self.closed = True
        self.handles.clear()
        self.seen.clear()
        self.origin_generations.clear()
        self.scopes.clear()
        self.scope_holds.clear()
        self.ledger = self.witness = self.journal = self.pipe = self.stream = None
        self.projected_input = None


def ledger_point(row: dict, available_frame: int | None) -> dict:
    """履歴の原時計を変えず、消費側で計算を利用可能になった時刻を別に渡す。"""
    if available_frame is None:
        return dict(frame_idx=row['frame_idx'], time_sec=row['time_sec'])
    if (type(available_frame) is not int or type(row['frame_idx']) is not int
            or available_frame < row['frame_idx'] or row['time_sec'] != row['frame_idx'] / FPS):
        raise ValueError('origin_history_available_clock')
    return dict(frame_idx=available_frame, time_sec=available_frame / FPS)
