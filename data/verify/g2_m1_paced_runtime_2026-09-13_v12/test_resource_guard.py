"""偽GPU票と元RAM境界。実温度の上昇を起こす試験ではない。"""
from __future__ import annotations
from pathlib import Path
import subprocess
from typing import Any
import pytest
import resource_guard as G


@pytest.mark.parametrize('raw', ['', '0, 50, Not Active', '1, 50, Not Active, Not Active',
    '0, NaN, Not Active, Not Active', '0, 50, Unknown, Not Active',
    '0, 50, Not Active, Not Active\n1, 40, Not Active, Not Active'])
def test_invalid_gpu(raw: str) -> None:
    with pytest.raises(ValueError): G.thermal(raw)


@pytest.mark.parametrize('case', ['normal', 'hw', 'sw', 'timeout', 'missing', 'rss', 'available', 'exact'])
def test_stop_conditions(monkeypatch: Any, case: str) -> None:
    rss = G.R.RSS_LIMIT_KIB + int(case == 'rss')
    available = G.R.AVAILABLE_MIN_KIB - int(case == 'available')
    monkeypatch.setattr(G.R, 'numbers', lambda path: {'VmRSS': rss, 'MemAvailable': available})
    def read() -> dict:
        if case == 'timeout': raise subprocess.TimeoutExpired('nvidia-smi', 3)
        if case == 'missing': return G.thermal('')
        return G.thermal('0, 50, ' + ('Active' if case == 'hw' else 'Not Active')
                         + ', ' + ('Active' if case == 'sw' else 'Not Active'))
    row = G.sample(Path('/proc/123'), read)
    assert row['safety_stop'] is (case not in ('normal', 'exact'))
    assert row['rss_kib'] == rss and row['available_kib'] == available
    assert row['cpu_package_temperature_c'] is None


def test_query_has_bounded_timeout(monkeypatch: Any) -> None:
    def run(command: list, **kwargs: Any) -> Any:
        assert command[0] == G.NVIDIA and kwargs['timeout'] == 3.0 and kwargs['check'] is True
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(G.subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired): G.gpu()
