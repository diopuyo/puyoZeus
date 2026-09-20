"""盤面未取得を欠測として保持し、開始資格とSTABLE履歴には採用しない。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
import anchor as A

SOURCE_SHA = '5e0ab0e36ff2f783a971d6fc48f5d552747d02d1beef6a37b8a2c671169b649a'
SIDES = A.SIDES
STABLE_CHECKS = ('stable', 'observed', 'raw_pixel_stable', 'persistence', 'board_present')


def checked_side(row: dict, side: Any, active_chain: Any) -> dict:
    args = row['arguments']
    grid = A.plain(args['board'])
    A.C.require((grid is None) == (side.confirmed_board is None), 'metadata_board_presence_mismatch')
    if grid is not None:
        result = A.checked_side(row, side, active_chain)
        result['checks']['board_present'] = True
        return result
    A.C.require(args['bstate']['value'] == side.state.value, 'metadata_state_mismatch')
    A.C.require(args['score'] == side.score and args['board_provenance'] == side.board_provenance, 'metadata_side_mismatch')
    A.C.require((args['chain_event_used_fields'] is None) == (side.chain_event is None), 'metadata_chain_mismatch')
    checks = dict(stable=side.state.value == 'stable', observed=side.board_provenance == 'observed',
        raw_pixel_stable=args['raw_pixel_stable'] is True, persistence=args['stable_persistence_confidence'] is True,
        score_zero=type(side.score) is int and side.score == 0, empty=False,
        no_chain=side.chain_event is None and active_chain is None, board_present=False)
    return dict(frame_idx=row['frame_idx'], side=row['side'], grid=None, arguments=deepcopy(args), checks=checks)


class Capture(A.Capture):
    def __init__(self, loop: Any, tail: Any, identity: dict[str, str]) -> None:
        A.C.require(hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest() == SOURCE_SHA, 'anchor_base_source')
        self.unavailable_boards: list[dict] = []
        super().__init__(loop, tail, identity)

    def completed(self, frame: int) -> None:
        previous_events = len(self.events)
        A.C.Capture.completed(self, frame)
        state = self.loop.runtime_state
        result, pipe = state['result'], state['pipeline']
        A.C.require(type(result.is_match_active) is bool, 'active_type')
        A.C.require(self.pipe is None or self.pipe is pipe, 'pipeline_changed')
        self.pipe = pipe
        sides = [checked_side(row, getattr(result, name), getattr(pipe, '_active_chain_' + label.lower()))
                 for row, name, label in zip(self.tail.rows, ('p1', 'p2'), SIDES, strict=True)]
        for side in sides:
            if not side['checks']['board_present']:
                self.unavailable_boards.append(dict(frame=frame, side=side['side'],
                    state=side['arguments']['bstate']['value'], reason='confirmed_board_unavailable'))
        self.boundary(frame, self.events[previous_events:])
        if result.is_match_active and all(all(s['checks'][k] for k in STABLE_CHECKS) for s in sides):
            self.stable_snapshots.extend(deepcopy(sides))
        if self.candidate is not None and self.anchor is None:
            self.qualify(frame, sides, result.is_match_active)

    def snapshot(self) -> dict:
        result = super().snapshot()
        result['unavailable_boards'] = deepcopy(self.unavailable_boards)
        return result


def install(stack: Any, loop: Any, tail: Any, identity: dict[str, str]) -> Capture:
    capture = Capture(loop, tail, identity)
    stack.callback(capture.close)
    return capture
