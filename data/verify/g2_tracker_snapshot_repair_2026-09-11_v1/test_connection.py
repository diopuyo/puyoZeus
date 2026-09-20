"""接続/一回装着/解除/終了保存の小対照。lease/原Jは人工で別検収。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import connection as C
import reproduce as R


def test_entry_close_save(tmp_path: Any) -> None:
    join, tracker = R.load_join(), R.tracker()
    module = N(parts=lambda: N(join=join))
    side = N(selected=lambda: module)
    original, leaf = side.selected, join._vrepr_leaf
    class Parent:
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            loaded = side.selected()
            assert loaded.parts().join is join
            self.state['snapshot'] = join._vrepr(tracker)
    with ExitStack() as stack:
        context = C.derived(Parent, side)()
        context.stack, context.pipe = stack, N(_chain_tracker_2p=tracker)
        context.state = {'output': tmp_path}
        context.perform(None, 34932, 582.2)
        assert side.selected is original and join._vrepr_leaf is not leaf
        with pytest.raises(AssertionError, match='repeated_perform'):
            context.perform(None, 34934, 582.3)
    assert join._vrepr_leaf is leaf
    C.save(tmp_path, context.state)
    result = json.loads((tmp_path / 'TRACKER_SNAPSHOT_REPAIR.json').read_text())
    assert result['restored'] and result['selected_restored'] and not result['G2']


def test_repeated_selected_rejected(tmp_path: Any) -> None:
    join, tracker = R.load_join(), R.tracker()
    side = N(selected=lambda: N(parts=lambda: N(join=join)))
    original = side.selected
    class Parent:
        def perform(self, *args: Any) -> None:
            side.selected()
            side.selected()
    with ExitStack() as stack:
        context = C.derived(Parent, side)()
        context.stack, context.pipe, context.state = stack, N(_chain_tracker_2p=tracker), {'output': tmp_path}
        with pytest.raises(AssertionError, match='second_selected'):
            context.perform(None, 0, 0.0)
        assert side.selected is original
    assert context.state['tracker_snapshot_repair']['restored']


def test_failure_saves_restoration(tmp_path: Any) -> None:
    join, tracker, failure = R.load_join(), R.tracker(), RuntimeError('failure')
    side = N(selected=lambda: N(parts=lambda: N(join=join)))
    class Parent:
        def perform(self, *args: Any) -> None:
            side.selected()
            raise failure
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            context = C.derived(Parent, side)()
            context.stack, context.pipe, context.state = stack, N(_chain_tracker_2p=tracker), {'output': tmp_path}
            context.perform(None, 0, 0.0)
    assert caught.value is failure
    result = json.loads((tmp_path / 'TRACKER_SNAPSHOT_REPAIR.json').read_text())
    assert result['restored'] and result['body_error'] == repr(failure)
