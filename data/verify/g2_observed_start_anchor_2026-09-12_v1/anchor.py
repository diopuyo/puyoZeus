"""実観測の開始条件を因果時刻へ束縛。映像の真値や本番資格は発行しない。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_producer_time_capture_2026-09-12_v1/capture.py'
BASE_SHA = '1454f3cd83cb1e70b84c3e839fee8542c6e3b9664a718d5d6370945c43ab646b'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA, 'capture_source'
spec = importlib.util.spec_from_file_location('_start_capture_base', BASE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
SIDES = ('1P', '2P')
ROWS, COLS = 13, 6


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict) and 'sequence_type' in value:
        return [plain(v) for v in value['items']]
    if isinstance(value, dict) and 'ndarray_dtype' in value:
        return plain(value['values'])
    return value


def checked_side(row: dict[str, Any], side: Any, active_chain: Any) -> dict[str, Any]:
    args = row['arguments']
    grid = plain(args['board'])
    C.require(type(grid) is list and len(grid) == ROWS and all(type(r) is list and len(r) == COLS for r in grid), 'grid_shape')
    C.require(grid == side.confirmed_board.to_dict()['grid'], 'metadata_grid_mismatch')
    C.require(args['bstate']['value'] == side.state.value, 'metadata_state_mismatch')
    C.require(args['score'] == side.score and args['board_provenance'] == side.board_provenance, 'metadata_side_mismatch')
    C.require((args['chain_event_used_fields'] is None) == (side.chain_event is None), 'metadata_chain_mismatch')
    checks = {'stable': side.state.value == 'stable', 'observed': side.board_provenance == 'observed',
              'raw_pixel_stable': args['raw_pixel_stable'] is True,
              'persistence': args['stable_persistence_confidence'] is True,
              'score_zero': type(side.score) is int and side.score == 0,
              'empty': all(type(v) is int and v == 0 for r in grid for v in r),
              'no_chain': side.chain_event is None and active_chain is None}
    return {'frame_idx': row['frame_idx'], 'side': row['side'], 'grid': grid,
            'arguments': deepcopy(args), 'checks': checks}


class Capture(C.Capture):
    def __init__(self, loop: Any, tail: Any, identity: dict[str, str]) -> None:
        C.require(set(identity) == {'source_id', 'run_id'}, 'start_identity_keys')
        self.candidate: dict[str, Any] | None = None
        self.anchor: dict[str, Any] | None = None
        self.decisions: list[dict[str, Any]] = []
        self.stable_snapshots: list[dict[str, Any]] = []
        self.game: int | None = None
        self.pipe: Any = None
        super().__init__(loop, tail, identity)

    def completed(self, frame: int) -> None:
        previous_events = len(self.events)
        super().completed(frame)
        state = self.loop.runtime_state
        result, pipe = state['result'], state['pipeline']
        C.require(type(result.is_match_active) is bool, 'active_type')
        C.require(self.pipe is None or self.pipe is pipe, 'pipeline_changed')
        self.pipe = pipe
        sides = [checked_side(row, getattr(result, name), getattr(pipe, '_active_chain_' + label.lower()))
                 for row, name, label in zip(self.tail.rows, ('p1', 'p2'), SIDES, strict=True)]
        additions = self.events[previous_events:]
        self.boundary(frame, additions)
        if result.is_match_active and all(all(s['checks'][k] for k in ('stable', 'observed', 'raw_pixel_stable', 'persistence')) for s in sides):
            self.stable_snapshots.extend(deepcopy(sides))
        if self.candidate is not None and self.anchor is None:
            self.qualify(frame, sides, result.is_match_active)

    def boundary(self, frame: int, additions: list[dict[str, Any]]) -> None:
        game = self.shared.game_idx
        C.require(type(game) is int and game >= 0, 'game_type')
        advances = [e for e in additions if e['list'] == 'advance_times']
        if self.game is not None and self.game != game:
            C.require(bool(advances), 'game_changed_without_advance')
        if advances:
            C.require(len(advances) == 1, 'multiple_advances')
            self.anchor = self.candidate = None
            matched = any(e['list'] == 'visual_rise_times' and e['raw_value'] == advances[0]['raw_value'] for e in additions)
            if matched:
                self.candidate = dict(identity=self.identity, game_idx=game, boundary_available_frame=frame,
                                      occurrence_sec=advances[0]['raw_value'], boundary_events=deepcopy(additions))
            else:
                self.decisions.append(dict(frame=frame, game_idx=game, eligible=False, reasons=['visual_boundary_absent']))
        self.game = game

    def qualify(self, frame: int, sides: list[dict[str, Any]], active: bool) -> None:
        reasons = [s['side'] + ':' + k for s in sides for k, ok in s['checks'].items() if not ok]
        if not active:
            reasons.append('match_inactive')
        pending = self.recorder._previous_pending
        C.require(type(pending) is tuple and len(pending) == len(SIDES), 'pending_shape')
        if any(type(v) is not int or v != 0 for v in pending):
            reasons.append('producer_pending_disagrees')
        evidence = dict(candidate=deepcopy(self.candidate), qualification_frame=frame,
                        metadata=deepcopy(list(self.tail.rows)), pending=list(pending))
        self.decisions.append(dict(frame=frame, game_idx=self.game, eligible=not reasons,
                                   reasons=reasons, raw_digest=digest(evidence)))
        if not reasons:
            self.anchor = dict(**evidence, evidence_sha256=digest(evidence),
                               classification='observed_start_conditions_not_video_ground_truth')

    def snapshot(self) -> dict[str, Any]:
        try:
            saved = super().snapshot()
            C.require(self.game == self.shared.game_idx, 'snapshot_game')
            saved.update(start_anchor=deepcopy(self.anchor), start_candidate=deepcopy(self.candidate),
                         start_decisions=deepcopy(self.decisions), stable_snapshots=deepcopy(self.stable_snapshots),
                         observed_game_idx=self.game, start_observation_eligible=self.anchor is not None)
            # 既存v1のgame_anchor_qualified/live_qualifiedはfalseのまま。実映像検収とは別。
            return saved
        except BaseException as error:
            self.error = self.loop.error = error
            raise


def install(stack: Any, loop: Any, tail: Any, identity: dict[str, str]) -> Capture:
    capture = Capture(loop, tail, identity)
    stack.callback(capture.close)
    return capture
