"""次手なしの人工シナリオで、旧二手geometryのcandidateだけを除く。実動画では使わない。"""
from __future__ import annotations
from dataclasses import asdict, replace
import json
from typing import Any


def install(stack: Any, pixels: Any, types: Any, reset: int, original_rows: list[Any], stream: Any) -> None:
    original = type(pixels).__call__
    def observed(self: Any, invocation: Any, side: str) -> Any:
        value = original(self, invocation, side)
        if self is not pixels or side != '1P' or invocation.frame < reset:
            return value
        assert type(value) is types.MotionObservation and value.invocation is invocation
        event = value.candidate
        result = replace(value, candidate=None)
        assert original_rows and original_rows[-1]['frame'] == value.frame_idx
        original_rows[-1].update(candidate_is_pre_transform=True, effective_candidate=None,
                                artificial_candidate_suppressed=event is not None)
        stream.write(json.dumps(dict(frame=value.frame_idx, side=side, artificial=True,
            scenario='basis_only_no_next_hand', source_candidate=None if event is None else asdict(event),
            effective_candidate=None, segment=value.segment_id, epoch=value.software_epoch,
            quiet=value.quiet_frames, invocation_preserved=result.invocation is invocation)) + '\n')
        return result
    type(pixels).__call__ = observed
    def restore() -> None:
        assert type(pixels).__call__ is observed, 'basis_only_input_foreign_restore'
        type(pixels).__call__ = original
    stack.callback(restore)
