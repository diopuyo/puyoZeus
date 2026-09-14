"""人工入力変換の原票/有効票、対象外非干渉、原例外保持を検査する。"""
from __future__ import annotations
import contextlib
import io
import json
from typing import Any
import pytest
import test_basis_input_contract as T
import basis_only_input as B


def test_no_next_hand_keeps_original_and_effective_evidence() -> None:
    rows: list[Any] = []
    original = T.view(T.I.FIRST)
    class Pixels:
        def __call__(self, invocation: Any, side: str) -> Any:
            rows.append(dict(frame=invocation.frame, candidate=vars(original.candidate)))
            return original
    pixels, stream = Pixels(), io.StringIO()
    old = Pixels.__call__
    with contextlib.ExitStack() as stack:
        B.install(stack, pixels, T.O, T.I.RESET, rows, stream)
        result = pixels(original.invocation, '1P')
        assert result.candidate is None and original.candidate is not None
        assert result.invocation is original.invocation and result.segment_id == original.segment_id
        assert result.quiet_frames == original.quiet_frames and result.software_epoch == original.software_epoch
    assert Pixels.__call__ is old
    assert rows[0]['candidate_is_pre_transform'] and rows[0]['artificial_candidate_suppressed']
    assert rows[0]['effective_candidate'] is None and rows[0]['candidate'] is not None
    assert json.loads(stream.getvalue())['source_candidate']['sequence_number'] == 1


@pytest.mark.parametrize('side,reset', [('2P', T.I.RESET), ('1P', T.I.SECOND)])
def test_other_side_and_prefix_are_not_transformed(side: str, reset: int) -> None:
    original = T.view(T.I.FIRST)
    class Pixels:
        def __call__(self, invocation: Any, side: str) -> Any:
            return original
    pixels, stream = Pixels(), io.StringIO()
    with contextlib.ExitStack() as stack:
        B.install(stack, pixels, T.O, reset, [], stream)
        assert pixels(original.invocation, side) is original
    assert not stream.getvalue()


def test_original_geometry_failure_is_not_hidden() -> None:
    error = ValueError('元の人工geometry失敗')
    class Pixels:
        def __call__(self, invocation: Any, side: str) -> Any:
            raise error
    pixels = Pixels()
    with contextlib.ExitStack() as stack:
        B.install(stack, pixels, T.O, T.I.RESET, [], io.StringIO())
        with pytest.raises(ValueError) as caught:
            pixels(T.view(T.I.FIRST).invocation, '1P')
    assert caught.value is error
