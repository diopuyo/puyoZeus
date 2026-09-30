"""Windows 性能コア優先 (live_win_cores): 解析・選択・既定OFFの不変・Windows 実機での取得。"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import pytest

from src.phase_j import live_win_cores as cores

FULL_ENTRY_BYTES = 32


def _entry(cpu_id: int, efficiency: int, kind: int = cores.CPU_SET_ENTRY_TYPE) -> bytes:
    body = cores.ENTRY_BODY.pack(cpu_id, 0, cpu_id % 256, cpu_id // 2 % 256, 0, 0, efficiency, 0)
    return (cores.ENTRY_HEADER.pack(FULL_ENTRY_BYTES, kind) + body).ljust(FULL_ENTRY_BYTES, b'\0')


def _blob(efficiencies: list[int]) -> bytes:
    return b''.join(_entry(0x100+index, value) for index, value in enumerate(efficiencies))


def test_parse_reads_every_entry_with_id_and_efficiency() -> None:
    blob = _blob([1, 1, 0, 0])
    assert cores.parse_cpu_sets(blob, len(blob)) == [(0x100, 1), (0x101, 1), (0x102, 0), (0x103, 0)]


def test_parse_skips_non_cpu_set_entries_but_keeps_walking() -> None:
    blob = _entry(1, 0, kind=7) + _entry(2, 1)
    assert cores.parse_cpu_sets(blob, len(blob)) == [(2, 1)]


@pytest.mark.parametrize('length_shift', [-1, 5])
def test_parse_rejects_broken_lengths(length_shift: int) -> None:
    blob = _blob([1, 0])
    truncated = blob[:len(blob)+length_shift] if length_shift < 0 else blob + struct.pack('<II', 4, 0)
    with pytest.raises(ValueError):
        cores.parse_cpu_sets(truncated, len(truncated))


def test_performance_ids_pick_top_class_only_on_hybrid() -> None:
    hybrid = [(10, 1), (11, 1), (12, 0), (13, 0)]
    assert cores.performance_cpu_set_ids(hybrid) == [10, 11]
    assert cores.performance_cpu_set_ids([(1, 0), (2, 0)]) == []   # 非ハイブリッド = 何もしない
    assert cores.performance_cpu_set_ids([]) == []


def test_default_is_off_and_env_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cores.PERFORMANCE_CORES_ENV, raising=False)
    assert cores.requested() is False
    monkeypatch.setenv(cores.PERFORMANCE_CORES_ENV, '1')
    assert cores.requested() is True


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows 以外の no-op を確認')
def test_non_windows_is_noop() -> None:
    assert cores.prefer_performance_cores() == []


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows 実機の CPU set 取得')
def test_windows_reads_real_cpu_sets_and_selection_is_subset() -> None:
    entries = cores.read_cpu_sets()
    assert len(entries) >= 1 and len({cpu for cpu, _ in entries}) == len(entries)
    chosen = cores.performance_cpu_set_ids(entries)
    assert set(chosen) <= {cpu for cpu, _ in entries}
    assert len(chosen) < len(entries) or not chosen   # 全部を選ぶことはない (選ぶなら一部)


def test_apply_runtime_snapshot_unchanged_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_cpu import apply_runtime
    monkeypatch.delenv(cores.PERFORMANCE_CORES_ENV, raising=False)
    assert 'performance_cpu_sets' not in apply_runtime('recognition')


def test_flag_propagates_to_env_and_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_live_pipeline_20260928 import add_fault_arguments, configure_cpu
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    parser = argparse.ArgumentParser()
    add_fault_arguments(parser)
    assert parser.parse_args([]).performance_cores is False
    on = parser.parse_args(['--performance-cores'])
    options = argparse.Namespace(cpu_threads=0, evaluation_nice=0, recognition_audit=False,
                                 worker_mode='process', source='video', **vars(on))
    configure_cpu(options, parser)
    assert os.environ[cores.PERFORMANCE_CORES_ENV] == '1'
