"""遅い PC で表示の遅れを抑える優先度の組み方 (live_priority、既定 OFF)。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.phase_j import live_cpu, live_priority

ROOT = Path(__file__).parents[2]
# 重い探索の子の入口 (spawn の target)。ここで apply_power_policy の直後に下げる。
BACKGROUND_ENTRIES = (('src/b3_async.py', '_compute'), ('src/b3_async.py', '_state_worker'),
                      ('src/phase_j/live_ex_async.py', '_compute'), ('src/phase_j/live_ex_async.py', '_serve'),
                      ('src/phase_j/live_ex_fire_first.py', 'serve'))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    monkeypatch.setattr(live_cpu, 'windows_priority', seen.append)
    monkeypatch.setattr(live_priority.sys, 'platform', 'win32')
    return seen


def test_default_off_keeps_background_priority(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    monkeypatch.delenv(live_priority.ENV, raising=False)
    assert live_priority.enabled() is False
    live_priority.lower_background()
    assert calls == []


def test_on_lowers_background_and_records(monkeypatch: pytest.MonkeyPatch, calls: list[int],
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(live_priority.ENV, '1')
    live_priority.lower_background()
    assert calls == [live_priority.WINDOWS_BELOW_NORMAL]
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record['event'] == live_priority.EVENT and record['status'] == 'applied'


def test_failure_is_recorded_and_streaming_continues(monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(live_priority.ENV, '1')
    monkeypatch.setattr(live_priority.sys, 'platform', 'win32')
    def refuse(priority: int) -> None:
        raise OSError('denied')
    monkeypatch.setattr(live_cpu, 'windows_priority', refuse)
    live_priority.lower_background()
    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])['status'] == 'failed'


@pytest.mark.parametrize('flag, role, lowered', [('0', 'evaluation', True), ('1', 'evaluation', False),
                                                 ('1', 'mc', True), ('1', 'event-evaluation', True)])
def test_evaluation_keeps_normal_only_when_on(monkeypatch: pytest.MonkeyPatch, flag: str, role: str,
                                              lowered: bool) -> None:
    monkeypatch.setenv(live_priority.ENV, flag)
    monkeypatch.setenv(live_cpu.NICE_ENV, '10')
    monkeypatch.setenv(live_cpu.THREAD_ENV, '0')
    seen: list[str] = []
    monkeypatch.setattr(live_cpu, 'windows_below_normal', lambda: seen.append('below'))
    def fake_nice(value: int) -> int:
        if value:
            seen.append('nice')
        return 0
    monkeypatch.setattr(live_cpu.os, 'nice', fake_nice, raising=False)  # 試験の process 自身の優先度は変えない
    if role == 'event-evaluation':
        monkeypatch.setattr('threadpoolctl.threadpool_limits', lambda limits: None)
    live_cpu.apply_runtime(role)
    assert bool(seen) is lowered


@pytest.mark.parametrize('path, function', BACKGROUND_ENTRIES)
def test_background_entries_lower_right_after_power_policy(path: str, function: str) -> None:
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function).body
    calls = [node.value.func.id for node in body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)]
    assert calls[:2] == ['apply_power_policy', 'lower_background']
