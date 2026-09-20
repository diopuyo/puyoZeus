"""凍結observerに対する親の別反例。実動画の品質検証ではない。"""
from __future__ import annotations
from contextlib import ExitStack
import copy
from pathlib import Path
from typing import Any
import pytest
import test_observer as T
import saved_context as S

setup = T.setup


@pytest.mark.parametrize("change", ("frame", "side", "postmutation"))
def test_actual_return_not_replaced_by_capture_failure(setup: Any, change: str) -> None:
    collector, pipe, history, state = setup
    original = collector.RecognitionPipeline.update
    def changed(self: Any, *args: Any, **kwargs: Any) -> Any:
        value = original(self, *args, **kwargs)
        if change == "frame":
            value.frame_idx += 2
        elif change == "side":
            value.p2.side = "1P"
        else:
            value.p1.confirmed_board[12][0] = 4
        return value
    collector.RecognitionPipeline.update = changed
    with ExitStack() as stack:
        rec = T.O.install(stack, collector, history, state, expected_frames=[T.FRAME])
        result = pipe.update(T.FRAME, T.FRAME / 60)
        assert result is pipe.last
    assert collector.RecognitionPipeline.update is changed
    if change != "postmutation":
        with pytest.raises(ValueError, match="lifetime_or_error"):
            T.O.finish(state)
    else:
        assert rec.rows[0]["sides"]["1P"]["before_hold"]["confirmed"] != rec.rows[0]["sides"]["1P"]["final"]["confirmed"]


def test_duplicate_update_no_extra_original_call(setup: Any) -> None:
    collector, pipe, history, state = setup
    with ExitStack() as stack:
        rec = T.O.install(stack, collector, history, state, expected_frames=[T.FRAME])
        first = pipe.update(T.FRAME, T.FRAME / 60)
        with pytest.raises(ValueError, match="duplicate"):
            pipe.update(T.FRAME, T.FRAME / 60)
        assert pipe.last is first
    assert len(rec.rows) == 1 and rec.errors
    with pytest.raises(ValueError, match="lifetime_or_error"):
        T.O.finish(state)


@pytest.mark.parametrize("code", (False, True, -1, 1))
def test_saved_complete_requires_actual_numeric_success(tmp_path: Path, code: Any) -> None:
    T.O.write(tmp_path / "COMPLETE", {"child_exit_code": code, "sha256": {}})
    with pytest.raises(S.B.ContextFault, match="real_child_exit"):
        S.verify_complete(tmp_path)


def test_saved_join_does_not_borrow_neighbor() -> None:
    row = {"frame_idx": 100, "sides": {side: dict.fromkeys(("pb", "sm", "next", "candidate_row")) for side in T.O.SIDES}}
    index = {key: {} for key in ("pb", "sm", "next", "candidate_row")}
    index["sm"][(98, "1P")] = {"frame_idx": 98}
    S.saved_join(row, index)
    row["sides"]["1P"]["sm"] = index["sm"][(98, "1P")]
    with pytest.raises(S.B.ContextFault, match="same_run_join"):
        S.saved_join(row, index)


def test_duplicate_json_key_never_last_value_wins() -> None:
    with pytest.raises(S.B.ContextFault, match="duplicate_json"):
        S.decode('{"match_end_locked":true,"match_end_locked":false}')
