"""CPU割当ては元の許可集合内で分離し、小型機のメイン経路を残す。"""
import pytest
from types import SimpleNamespace
from src.phase_j.live_obs_cpu import split_mask


@pytest.mark.parametrize('allowed,main,ama', [
    (1, 1, 1), (3, 3, 3), (15, 7, 8), (255, 63, 192),
    (65535, 16383, 49152), (61440, 28672, 32768), (85, 21, 64),
])
def test_split_mask(allowed: int, main: int, ama: int) -> None:
    assert split_mask(allowed) == (main, ama)
    assert main | ama == allowed
    if allowed.bit_count() >= 4:
        assert not main & ama
        assert main.bit_count() >= 3


def test_idle_only_changes_native_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_obs_cpu as cpu
    calls = []
    kernel = SimpleNamespace(GetCurrentProcess=lambda: 456,
        SetPriorityClass=lambda handle, priority: calls.append((handle, priority)) or 1)
    monkeypatch.setattr(cpu, 'kernel_api', lambda: kernel)
    monkeypatch.setattr(cpu, 'os', SimpleNamespace(name='nt', environ={
        'PUYO_OBS_AMA_IDLE': '1', 'PUYO_OBS_CPU_ISOLATION': '0', 'PUYO_OBS_AMA_MASK': '128'}))
    cpu.configure_ama(123)
    cpu.configure_ama()
    assert calls == [(123, cpu.IDLE), (456, cpu.BELOW_NORMAL)]
