"""原PuyoGroupだけ全値で比較する。汎用深さ上限と他の検査は変えない。"""
from __future__ import annotations
import dataclasses
import sys
from types import FunctionType
from typing import Any

FIELDS = ('color', 'cells', 'size', 'ojama_adjacent')
ROWS, COLS = 13, 6


def group_type(pipe: Any) -> Any:
    simulator = pipe._chain_tracker_2p._simulator
    module = sys.modules[type(simulator).__module__]
    value = module.PuyoGroup
    assert value is module.ChainSimulator.simulate.__globals__['PuyoGroup']
    assert tuple(item.name for item in dataclasses.fields(value)) == FIELDS
    assert value.__dataclass_params__.frozen
    return value


def points(value: Any) -> Any:
    assert type(value) is frozenset, 'group_points_type'
    assert len(value) <= ROWS * COLS, 'group_points_count'
    for cell in value:
        assert type(cell) is tuple and len(cell) == 2, 'group_cell_shape'
        assert all(type(number) is int for number in cell), 'group_cell_type'
        assert 0 <= cell[0] < ROWS and 0 <= cell[1] < COLS, 'group_cell_range'
    return tuple(sorted(value))


def leaf(original: Any, kind: Any, error_type: Any = ValueError) -> Any:
    def render(value: Any) -> Any:
        if type(value) is not kind:
            return original(value)
        try:
            assert set(vars(value)) == set(FIELDS), 'group_extra_fields'
            assert type(value.color) is int and type(value.size) is int, 'group_scalar_type'
            cells, ojama = points(value.cells), points(value.ojama_adjacent)
            assert value.size == len(cells), 'group_size_mismatch'
            return ('qualified_PuyoGroup', id(kind), value.color, cells, value.size, ojama)
        except (AssertionError, TypeError, AttributeError) as error:
            raise error_type('PuyoGroup:' + str(error)) from error
    return render


def install(stack: Any, join: Any, kind: Any, evidence: dict[str, Any]) -> None:
    original = join._vrepr_leaf
    assert isinstance(original, FunctionType) and original.__globals__ is vars(join), 'snapshot_foreign_leaf'
    assert original.__code__.co_name == '_vrepr_leaf', 'snapshot_already_installed'
    assert original.__code__.co_filename == join._vrepr.__code__.co_filename, 'snapshot_foreign_source'
    wrapper = leaf(original, kind, join.SideJoinUninspectable)
    evidence.update(installed=True, restored=False, original_depth=join._MAX_DEPTH,
                    exact_type_module=kind.__module__, exact_type_id=id(kind))
    def close(kind: Any, body: Any, trace: Any) -> bool:
        try:
            assert join._vrepr_leaf is wrapper, 'snapshot_foreign_override'
            join._vrepr_leaf = original
            evidence['restored'] = True
        except BaseException as error:
            evidence['cleanup_error'] = repr(error)
            if body is None:
                raise
        return False
    stack.push(close)
    join._vrepr_leaf = wrapper
