"""6GiB留保と実測前の単独起動を確認する。"""
import pytest

from scripts.image_free_resources import capacity
from scripts.image_free_parallel import GIB, RESERVE_BYTES


@pytest.mark.parametrize('free,expected', [(5*GIB, 0), (7*GIB, 0), (10*GIB, 1)])
def test_unmeasured_starts_one(free: int, expected: int) -> None:
    assert capacity(free, 0, 0, 24, 0) == expected


def test_memory_limits_workers() -> None:
    assert capacity(RESERVE_BYTES+5*GIB, 0, GIB, 24, 0) == 4


def test_accounts_existing_workers() -> None:
    assert capacity(RESERVE_BYTES+3*GIB, 2, GIB, 24, 2) == 4


def test_external_cpu_load_limits_workers() -> None:
    assert capacity(100*GIB, 2, GIB, 24, 20) == 6


def test_below_reserve_does_not_admit_workers() -> None:
    assert capacity(5*GIB, 0, GIB, 24, 0) == 0
