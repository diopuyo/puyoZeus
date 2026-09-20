"""非所有の確定PB診断が未確率/多候補を丸めないことを検査する。"""
from copy import deepcopy
import pytest
from probe_opponent_projection import deterministic_grid


def probability() -> dict:
    return dict(present=True, type_valid=True, errors=[],
                cells=[[[[0, 1.0]] for _ in range(6)] for _ in range(13)])


def test_deterministic_snapshot_is_readonly() -> None:
    value = probability()
    before = deepcopy(value)
    assert deterministic_grid(value) == tuple((0,) * 6 for _ in range(13))
    assert value == before


@pytest.mark.parametrize('cell', [[], [[0, .7], [1, .3]], [[0, .9]], [[10, 1.0]]])
def test_missing_uncertain_or_unknown_is_not_invented(cell: list) -> None:
    value = probability()
    value['cells'][0][0] = cell
    with pytest.raises(AssertionError):
        deterministic_grid(value)
