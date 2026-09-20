"""実保存票を私有SplitOwnerへ逐次接続。物理認証・本番権・旧債務精算はしない。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

UNIT = Path(__file__).resolve().parent
PROJECT = UNIT.parents[2]
DIAG = PROJECT / 'data/verify/g2_inventory_producer_diagnosis_2026-09-09_v1/probe.py'
SPLIT = PROJECT / 'data/verify/g2_current_accounting_split_2026-09-08_v1/split_contract.py'
FIXED = {DIAG: 'f3ea5e870d344436739e3ea706b39fcec58112d8fe79b1cc08c0739e4e1dc244',
         SPLIT: 'd9c90ce7edf12051f1def730c641354fe0be1df00c6cf6adada66191462660c1'}
START_FRAME, OPERATIONS_PER_FRAME = 32684, 4
GRID_ROWS, GRID_COLS = 13, 6
RAW_UNKNOWN = 10
TEST_TAPES: dict[str, tuple[Any, ...]] | None = None


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def load(name: str, path: Path) -> Any:
    require(hashlib.sha256(path.read_bytes()).hexdigest() == FIXED[path], 'dependency_sha')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = load('_machine_inventory_diagnostic', DIAG)
S = load('_machine_inventory_split', SPLIT)


@dataclass(frozen=True)
class Witness:
    context: str
    frame_side: str
    accounting: str
    context_line: int
    frame_line: int
    origin: str
    context_file_sha256: str
    frames_file_sha256: str

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def values(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return json.loads(self.context), json.loads(self.frame_side), json.loads(self.accounting)


def extract(w: Witness, side: str) -> dict[str, Any]:
    row, frame, account = w.values()
    final = row['sides'][side]['final']
    require(frame['side'] == side, 'witness_side')
    require(row['frame_idx'] == frame['frame_idx'] == account['frame_idx'], 'witness_clock')
    require(row['time_sec'] == frame['time_sec'] == account['time_sec'], 'witness_time')
    require(account['side'] == side, 'accounting_side')
    require(row['update']['returned_frame_idx'] == row['available_frame'] == row['frame_idx'], 'return_frame')
    require(row['update']['returned_time_sec'] == row['time_sec'], 'return_time')
    capture = frame['accounting_capture']
    require(frame['raw_captured_this_frame'] is True and capture['captured_frame'] == row['frame_idx'], 'raw_capture')
    require(D.grid(final['confirmed']) == D.grid(frame['confirmed']), 'final_frame_grid')
    unknown = raw_unknown(capture['raw'])
    raw, current = D.grid(capture['raw']), D.grid(final['confirmed'])
    require(current is not None, 'missing_or_nonpermitted_confirmed')
    pb = D.pointmass(final['probability'])
    return {'row': row, 'frame': row['frame_idx'], 'time': row['time_sec'],
            'raw': raw, 'grid': current, 'pb': pb, 'state': final['state'],
            'epoch': D.epoch(row, side), 'before_epoch': D.epoch(row, side, 'before'),
            'source': row['source_id'], 'run': row['run_id'], 'sha': w.sha256,
            'matched': current == pb == raw, 'accounting': account,
            'next_pair': final['next_pair'], 'dnext_pair': final['dnext_pair'],
            'raw_unknown': unknown}


def raw_unknown(value: dict[str, Any]) -> list[tuple[int, int]]:
    raw = value.get('grid')
    require(type(raw) is list and len(raw) == GRID_ROWS, 'missing_or_malformed_raw')
    require(all(type(r) is list and len(r) == GRID_COLS for r in raw), 'malformed_raw_shape')
    require(all(type(c) is int and c in D.VALID | {RAW_UNKNOWN} for r in raw for c in r),
            'nonpermitted_raw_cell')
    return [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS) if raw[r][c] == RAW_UNKNOWN]


def clock(value: dict[str, Any], operation: int = 0) -> Any:
    require(0 <= operation < OPERATIONS_PER_FRAME, 'operation_order')
    return S.Clock(value['frame'], value['time'], value['frame'] * OPERATIONS_PER_FRAME + operation)


class BoundPolicy:
    """検証済み原票・直前私有stateへ完全束縛。一回の閉じたCPU許可のみ。"""

    def __init__(self) -> None:
        self._permit: tuple[str, str, str] | None = None
        self.proofs: list[dict[str, Any]] = []

    def arm(self, purpose: str, evidence: Any, state: Any, proof: dict[str, Any]) -> None:
        require(self._permit is None and purpose in ('baseline', 'placement'), 'policy_busy_or_purpose')
        require(evidence.evidence_id == digest(proof) if purpose == 'baseline'
                else evidence.event_id == digest(proof), 'proof_id')
        self._permit = (purpose, encoded(asdict(evidence)), encoded(asdict(state)))
        self.proofs.append({'purpose': purpose, 'proof': proof, 'proof_sha': digest(proof)})

    def authorize(self, purpose: str, evidence: Any, state: Any) -> None:
        require(self._permit == (purpose, encoded(asdict(evidence)), encoded(asdict(state))), 'unbound_evidence')
        self._permit = None


class InventoryProducer:
    """一つのprivate interval。停止後は保存するだけでownerを更新しない。"""

    def __init__(self, tape: tuple[Witness, ...], side: str) -> None:
        require(side in D.SIDES, 'side')
        require(type(tape) is tuple and bool(tape), 'unconnected_evidence_catalog')
        first = tape[0]
        self._catalog = frozenset(w.sha256 for w in tape)
        value = extract(first, side)
        self.side, self.source, self.run = side, value['source'], value['run']
        self.reset, self.last_frame = value['epoch'][0], value['frame']
        self.last_time, self.origin = value['time'], first.origin
        self._healthy(value)
        require(value['state'] == 'STABLE' and value['matched'], 'baseline_not_full_pointmass')
        require(value['epoch'][1] in (None, 0), 'baseline_action_already_progressed')
        namespace = f'private-software-interval:{digest([self.run, side, self.reset, self.last_frame])}'
        scope = S.Scope(self.source.removeprefix('sha256:'), self.run, side, namespace, self.reset)
        self.policy = BoundPolicy()
        self._owner = S.SplitOwner(scope, self.policy)
        self.inventory_grid = value['grid']
        self.records, self.pending, self.stopped = [], None, None
        self.last_match = value
        self.witnesses = [first]
        proof = {'kind': 'baseline', 'witness': asdict(first), 'side': side,
                 'private_scope': asdict(scope), 'counts': S.color_counts(value['grid'])}
        evidence = S.BaselineEvidence(scope, clock(value), S.color_counts(value['grid']), digest(proof), clock(value))
        self.policy.arm('baseline', evidence, self._owner.state, proof)
        self._owner.establish_baseline(evidence, clock(value))
        self._record(value, 'baseline', before=None)

    def _healthy(self, value: dict[str, Any]) -> None:
        require(value['source'] == self.source and value['run'] == self.run, 'source_run_changed')
        require(value['epoch'][0] == self.reset and value['before_epoch'][0] == self.reset, 'reset_changed')
        require(D.controlled(value['row']), 'active_lock_or_capture_failure')
        upstream = value['row'].get('upstream_failures')
        require(upstream == {name: [] for name in ('hidden_probability_observer', 'current_scope_sink',
                                                  'provisional_current_connection')}, 'upstream_failure')
        require(value['state'] in ('STABLE', 'TSUMO_FALL'), 'state_interrupt:' + value['state'])

    def _record(self, value: dict[str, Any], outcome: str, before: Any) -> None:
        self.records.append({'frame': value['frame'], 'time': value['time'], 'outcome': outcome,
             'witness_sha': value['sha'], 'epoch': value['epoch'], 'state': value['state'],
             'grid_sha': D.grid_sha(value['grid']), 'matched': value['matched'],
             'private_counter_before': before, 'private_counter_after': self._owner.state.counter,
             'revision': self._owner.state.counter_revision, 'software_action': self._owner.state.action,
             'old_counter': value['accounting']['after']['tsumo_count'],
             'old_counter_written': False, 'next_pair': value['next_pair'], 'dnext_pair': value['dnext_pair'],
             'raw_unknown_positions': value['raw_unknown'], 'raw_unknown_not_inventory_evidence': True})

    def _begin_fall(self, value: dict[str, Any]) -> None:
        action = value['epoch'][1]
        require(type(action) is int and action == self._owner.state.action + 1, 'software_action_gap')
        require(value['grid'] == self.inventory_grid, 'fall_baseline_discontinuity')
        self._owner.advance_action(action, clock(value))
        self.pending = {'start': value, 'anchor': self.last_match,
                        'start_index': len(self.witnesses) - 1, 'first_complete': None,
                        'first_incomplete_stable': None}

    def _complete(self, value: dict[str, Any], delta: dict[str, Any]) -> str:
        hits = D.placement_matches(self.inventory_grid, value['grid'], delta['pair'])
        if not hits:
            return 'hold_unlanded_geometry'
        if self.pending['first_complete'] is None:
            self.pending['first_complete'] = value
        if not value['matched']:
            return 'hold_completion_evidence'
        proof = self._placement_proof(value, delta, hits)
        occurred = clock(self.pending['first_complete'])
        evidence = S.PlacementEvidence(self._owner.state.scope, digest(proof), value['epoch'][1],
                   clock(value, 1), tuple(delta['pair'].count(c) for c in range(1, 6)), occurred)
        self.policy.arm('placement', evidence, self._owner.state, proof)
        self._owner.add_placement(evidence, clock(value, 1))
        self.inventory_grid, self.last_match, self.pending = value['grid'], value, None
        return 'placement_added'

    def _placement_proof(self, value: dict[str, Any], delta: dict[str, Any], hits: list[Any]) -> dict[str, Any]:
        start, anchor = self.pending['start'], self.pending['anchor']
        return {'kind': 'placement', 'side': self.side, 'source': self.source, 'run': self.run,
                'scope': asdict(self._owner.state.scope), 'software_action': value['epoch'][1],
                'anchor_frame': anchor['frame'], 'anchor_sha': anchor['sha'],
                'fall_started_at': asdict(clock(start)),
                'occurred_observation_interval': [start['frame'] - D.STRIDE,
                                                  self.pending['first_complete']['frame']],
                'occurred_is_observed_completion_not_physical_time': True,
                'available_at': asdict(clock(value, 1)), 'before_grid': self.inventory_grid,
                'after_grid': value['grid'], 'observed_added': delta['pair'], 'placements': hits,
                'witnesses': [asdict(w) for w in self.witnesses[self.pending['start_index']:]],
                'counter_before': self._owner.state.counter, 'history_revision': self._owner.state.counter_revision,
                'pair_provenance': 'full_grid_addition_not_physical_tsumo_identity'}

    def _step(self, value: dict[str, Any]) -> str:
        delta = D.difference(self.inventory_grid, value['grid'])
        require(delta['kind'] not in ('baseline_cell_changed', 'garbage_change', 'other_increase'),
                'grid_discontinuity:' + delta['kind'])
        if self.pending is None:
            if value['state'] == 'TSUMO_FALL':
                self._begin_fall(value)
                return 'fall_started_without_addition'
            require(delta['kind'] == 'same_grid', 'stable_change_without_fall')
            require(value['epoch'][1] in (None, self._owner.state.action), 'action_changed_without_fall')
            if value['matched']:
                self.last_match = value
            return 'same_inventory_observed'
        require(value['epoch'][1] == self._owner.state.action, 'unresolved_action_before_next')
        if value['state'] != 'STABLE':
            return 'fall_observed_without_addition'
        if delta['kind'] != 'two_color_addition':
            if self.pending['first_incomplete_stable'] is None:
                self.pending['first_incomplete_stable'] = value['frame']
            return 'hold_incomplete_' + delta['kind']
        return self._complete(value, delta)

    def consume(self, witness: Witness) -> str:
        require(self.stopped is None, 'prefix_already_stopped')
        before = self._owner.state.counter
        try:
            require(witness.sha256 in self._catalog, 'unregistered_witness_hash')
            value = extract(witness, self.side)
            require(witness.origin == self.origin, 'witness_origin_changed')
            require(value['frame'] == self.last_frame + D.STRIDE, 'missing_or_duplicate_frame')
            require(value['time'] > self.last_time, 'time_not_forward')
            self._healthy(value)
            self.witnesses.append(witness)
            outcome = self._step(value)
            self.last_frame, self.last_time = value['frame'], value['time']
            self._record(value, outcome, before)
            return outcome
        except (ValueError, KeyError, TypeError) as error:
            self.stopped = {'reason': str(error), 'witness_sha': witness.sha256,
                            'frame': json.loads(witness.context).get('frame_idx'),
                            'private_counter': self._owner.state.counter}
            return 'stopped'

    def snapshot(self) -> dict[str, Any]:
        value = {'owner': asdict(self._owner.state), 'inventory_grid': self.inventory_grid,
                'records': self.records, 'proofs': self.policy.proofs, 'stopped': self.stopped,
                'first_unresolved_stable': self.pending['first_incomplete_stable'] if self.pending else None,
                'envelope': {'private_internal_accounting_available': self._owner.state.accounting_available,
                 'public_accounting_available': False, 'baseline_before_old_debt': 'UNKNOWN',
                 'physical_game_certified': False, 'physical_occurrence_certified': False,
                 'quality_gate_clear': False, 'live_permission': False,
                 'interval_kind': 'private_software_inventory_interval', 'source_origin': self.origin}}
        return json.loads(encoded(value))


def saved_tapes() -> tuple[dict[str, tuple[Witness, ...]], dict[str, str]]:
    hashes, receipt = D.verify_inputs()
    require(D.sha(DIAG) == FIXED[DIAG] and D.sha(SPLIT) == FIXED[SPLIT], 'frozen_source')
    sys.path.insert(0, str(D.SNAPSHOT))
    frames, _ = D.frame_index()
    tapes = {s: [] for s in D.SIDES}
    with (D.LIVE / 'provisional_context.jsonl').open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            require(row['source_id'] == receipt['source_id'] and row['run_id'] == receipt['run_id'], 'tape_source')
            if row['frame_idx'] < START_FRAME:
                continue
            for side in D.SIDES:
                frame = frames[(row['frame_idx'], side)]
                number, saved = frame['frame_side']
                accounting = frame['accounting_update'][1]
                tapes[side].append(Witness(encoded(row), encoded(saved), encoded(accounting),
                    line_number, number, 'actual_saved_video38_context_and_frames',
                    hashes[str(D.LIVE / 'provisional_context.jsonl')], hashes[str(D.LIVE / 'frames.jsonl')]))
    hashes[str(DIAG)], hashes[str(SPLIT)] = FIXED[DIAG], FIXED[SPLIT]
    return {s: tuple(values) for s, values in tapes.items()}, hashes


def replay_prefix(tape: tuple[Witness, ...], side: str) -> InventoryProducer:
    require(bool(tape), 'empty_tape')
    producer = InventoryProducer(tape, side)
    for witness in tape[1:]:
        if producer.consume(witness) == 'stopped':
            break
    if producer.stopped is None and producer.pending is not None:
        producer.stopped = {'reason': 'prefix_end_completion_unconfirmed', 'frame': producer.last_frame,
                            'private_counter': producer._owner.state.counter}
    return producer
