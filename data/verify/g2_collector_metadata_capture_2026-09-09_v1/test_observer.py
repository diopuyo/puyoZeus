"""実frozen採録関数のmetadata・既定OFF・例外復元を検査する。"""
from __future__ import annotations
import ast
import contextlib
import copy
from dataclasses import fields
import functools
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import pytest
import observer as O


@pytest.fixture(scope='module')
def collector() -> Any:
    path = O.PROJECT / 'scripts/diagnose_video38_confirmed_collapse_v1.py'
    spec = importlib.util.spec_from_file_location('_metadata_frozen_loader', path)
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    value = base.load_collector()
    before = {str(Path(m.__file__).resolve()): O.sha(Path(m.__file__)) for n, m in sys.modules.items()
        if n.startswith('src.') and getattr(m, '__file__', None)}
    yield value
    assert before == {p: O.sha(Path(p)) for p in before}


def call(c: Any, acc: Any, state: Any, shared: Any, *, tag: Any, next_pair: Any) -> None:
    grid = [[0] * 6 for _ in range(13)]
    grid[12][0] = 1
    c._process_side_lean(acc, state, '1P', c.Board.from_list(grid), c.BoardState.STABLE,
        90, 'artificial-metadata', O.FIRST / O.FPS, O.FIRST, next_pair=next_pair,
        dnext_pair=None, shared_game=shared, exclude_phantom=True, tsumo_count=2,
        all_clear_pending=1, ojama_net_balance=-3.0, ojama_forecast=4.0,
        match_end_locked=False, post_match_lockdown_active=False, raw_pixel_stable=bool(tag),
        stable_persistence_confidence=tag, board_provenance='observed')


def stored(acc: Any) -> dict[str, Any]:
    return {f.name: O.serial(getattr(acc, f.name)) for f in fields(acc) if f.name != 'wons'}


@pytest.mark.parametrize('tag,next_pair', ((True, (1, 2)), (False, (9, 1)), (None, None)))
def test_real_append_all_metadata_noninterference(collector: Any, tmp_path: Path, tag: Any, next_pair: Any) -> None:
    c, collected, histories = collector, [], []
    original = c._process_side_lean
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        output.mkdir()
        state = {'output': output}
        acc, side, shared = c._LeanNpzAccumulator(), c._SideState(), c._SharedGameCounter()
        with contextlib.ExitStack() as stack:
            O.install(stack, c, None, state, enabled=enabled)
            call(c, acc, side, shared, tag=tag, next_pair=next_pair)
        collected.append((stored(acc), O.serial(side), O.serial(shared)))
        assert 'append' not in vars(acc) and c._process_side_lean is original and len(acc.grids) == 1
        if enabled:
            sink = state[O.KEY]
            assert sink.closed and not sink.errors and len(sink.rows) == 1
            row = sink.rows[0]
            assert len(row['appends']) == 1 and row['arguments']['stable_persistence_confidence'] is tag
            assert O.validate_row(row, O.FIRST, '1P', 0) == 1
            actual = row['appends'][0]['stored_nonlabel_row']
            assert set(actual) == set(stored(acc)) and 'wons' not in actual
            assert actual['stable_persistence_confidences'] == (-1 if tag is None else int(tag))
            assert actual['game_idxs'] == side.game_idx and actual['all_clear_pendings'] == 1
            assert actual['next1_as'] == (-1 if next_pair is None else next_pair[0])
            histories = [json.loads(line) for line in (output / O.ENTRIES).read_text().splitlines()]
    assert collected[0] == collected[1] and len(histories) == 1


@pytest.mark.parametrize('mutation', ('clock', 'tag', 'game', 'label', 'sequence', 'append_grid', 'float_frame', 'bool_game'))
def test_saved_metadata_corruption_rejected(collector: Any, tmp_path: Path, mutation: str) -> None:
    c, state = collector, {'output': tmp_path}
    with contextlib.ExitStack() as stack:
        O.install(stack, c, None, state, enabled=True)
        call(c, c._LeanNpzAccumulator(), c._SideState(), c._SharedGameCounter(), tag=True, next_pair=(1, 2))
    row = copy.deepcopy(state[O.KEY].rows[0])
    item = row['appends'][0]
    if mutation == 'clock':
        row['frame_idx'] += O.STRIDE
    elif mutation == 'tag':
        item['stored_nonlabel_row']['stable_persistence_confidences'] = 0
    elif mutation == 'game':
        item['stored_nonlabel_row']['game_idxs'] += 1
    elif mutation == 'label':
        item['stored_nonlabel_row']['wons'] = False
    elif mutation == 'sequence':
        item['before_count'] = 1
    elif mutation == 'float_frame':
        row['frame_idx'] = float(row['frame_idx'])
    elif mutation == 'bool_game':
        item['stored_nonlabel_row']['game_idxs'] = False
    else:
        next(p for p in item['arguments']['mapping'] if p[0] == 'grid')[1] = None
    with pytest.raises(ValueError):
        O.validate_row(row, O.FIRST, '1P', 0)


def test_saved_empty_scope_rejected(tmp_path: Path) -> None:
    (tmp_path / O.ENTRIES).touch()
    with pytest.raises(ValueError, match='missing_rows'):
        O.saved_counts(tmp_path)


def test_original_exception_identity_and_instance_restore(collector: Any, tmp_path: Path) -> None:
    c, error = collector, RuntimeError('人工原採録例外')
    sink = O.Sink(c, tmp_path)
    original = c._process_side_lean
    @functools.wraps(original)
    def failing(*args: Any, **kwargs: Any) -> None:
        raise error
    c._process_side_lean = sink.wrapper(failing)
    acc = c._LeanNpzAccumulator()
    try:
        with pytest.raises(RuntimeError) as caught:
            call(c, acc, c._SideState(), c._SharedGameCounter(), tag=False, next_pair=None)
        assert caught.value is error and sink.errors and not sink.busy and 'append' not in vars(acc)
    finally:
        c._process_side_lean = original
        sink.close()


def test_disabled_and_invalid_source(collector: Any) -> None:
    with contextlib.ExitStack() as stack:
        O.install(stack, object(), None, {}, enabled=False)
        with pytest.raises(ValueError):
            O.install(stack, collector, None, {O.KEY: object()}, enabled=True)


def test_function_lengths() -> None:
    for name in O.OWN:
        if name.endswith('.py'):
            for node in ast.walk(ast.parse((O.ROOT / name).read_bytes())):
                if isinstance(node, ast.FunctionDef):
                    assert node.end_lineno - node.lineno + 1 <= 50, (name, node.name)
