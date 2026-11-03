"""二重起動の検出 (名前付きミューテックス / flock)・port の排他 bind 試行・停止の案内を検査する。

鍵の名前は試験ごとに一意にし (実機の PuyoLive や他の試験と衝突しない)、保持した鍵は必ず解放する。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from src.phase_j import live_instance_guard as guard

ROOT = Path(__file__).resolve().parents[1]
HOLDER_SCRIPT = (
    'import sys\n'
    'from src.phase_j import live_instance_guard as g\n'
    'g.MUTEX_NAME_PREFIX = sys.argv[1]\n'
    'g.LOCK_DIR_NAME = sys.argv[2]\n'
    'lock = g.acquire_lock(g.InstanceKey("port", sys.argv[3], "port"))\n'
    'print("owned" if lock.owned else "not-owned", flush=True)\n'
    'sys.stdin.readline()\n')


@pytest.fixture(autouse=True)
def isolated_names(monkeypatch: pytest.MonkeyPatch) -> str:
    """試験ごとに一意な鍵の名前空間。process 内に保持済みの鍵も空にする。"""
    token = uuid.uuid4().hex[:12]
    monkeypatch.setattr(guard, 'MUTEX_NAME_PREFIX', f'Local\\PuyoLiveTest-{token}-')
    monkeypatch.setattr(guard, 'LOCK_DIR_NAME', f'puyo_live_locks_test_{token}')
    monkeypatch.setattr(guard, '_HELD', [])
    monkeypatch.delenv(guard.ALLOW_SECOND_INSTANCE_ENV, raising=False)
    return token


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def make_keys(port: int, index: int = 3) -> list[guard.InstanceKey]:
    return guard.instance_keys('dshow', '127.0.0.1', port, index)


def test_keys_cover_port_and_camera_for_device_input_only() -> None:
    device = guard.instance_keys('dshow', '127.0.0.1', 8765, 3)
    assert [key.kind for key in device] == ['port', 'camera']
    assert device[0].name == 'port-127_0_0_1-8765' and device[1].name == 'camera-3'
    assert '8765' in device[0].label and 'index 3' in device[1].label
    video = guard.instance_keys('video', '127.0.0.1', 8765, 0)
    assert [key.kind for key in video] == ['port']          # 動画入力 (開発用) はカメラを占有しない
    assert guard.instance_keys('dshow', '127.0.0.1', 8766, 3)[0].name != device[0].name
    assert guard.instance_keys('dshow', '127.0.0.1', 8765, 4)[1].name != device[1].name


def test_second_acquire_of_same_key_is_not_owned_until_released() -> None:
    key = make_keys(free_port())[0]
    first = guard.acquire_lock(key)
    second = guard.acquire_lock(key)
    try:
        assert first.owned and not second.owned
    finally:
        guard.release_lock(second)
        guard.release_lock(first)
    third = guard.acquire_lock(key)
    try:
        assert third.owned            # 解放後は取り直せる (process 終了でも OS が同様に解放する)
    finally:
        guard.release_lock(third)


def test_other_process_holding_the_key_is_seen_as_not_owned(isolated_names: str) -> None:
    """実際の別 process が持っている鍵を検出し、その process が終わると取れる (kernel 資源の実測)。"""
    key = make_keys(free_port())[0]
    holder = subprocess.Popen(
        [sys.executable, '-c', HOLDER_SCRIPT, guard.MUTEX_NAME_PREFIX, guard.LOCK_DIR_NAME, key.name],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'owned'
        mine = guard.acquire_lock(key)
        assert not mine.owned
        guard.release_lock(mine)
    finally:
        holder.stdin.write('\n')
        holder.stdin.flush()
        holder.wait(timeout=30)
    after = guard.acquire_lock(key)
    try:
        assert after.owned
    finally:
        guard.release_lock(after)


def test_acquire_instances_reports_conflict_after_grace_and_polls_meanwhile() -> None:
    keys = make_keys(free_port())
    held = [guard.acquire_lock(key) for key in keys]
    sleeps: list[float] = []
    clock = [0.0]
    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds
    try:
        result = guard.acquire_instances(keys, grace_sec=1.0, clock=lambda: clock[0], sleep=sleep)
        assert [key.kind for key in result.conflicts] == ['port', 'camera']
        assert result.held == [] and result.errors == []
        assert len(sleeps) >= 2 * 3          # 猶予の間、各鍵を取り直し続けた (1.0 / 0.25 秒)
    finally:
        for lock in held:
            guard.release_lock(lock)


def test_acquire_instances_succeeds_when_previous_holder_exits_within_grace() -> None:
    """再起動の直後で前の process がまだ終了処理中でも、猶予内に解放されれば二重起動と見なさない。"""
    keys = make_keys(free_port())
    held = [guard.acquire_lock(key) for key in keys]
    clock = [0.0]
    def sleep(seconds: float) -> None:
        clock[0] += seconds
        while held:
            guard.release_lock(held.pop())      # 1 回待ったところで前の process が終わる
    result = guard.acquire_instances(keys, grace_sec=3.0, clock=lambda: clock[0], sleep=sleep)
    try:
        assert result.conflicts == [] and len(result.held) == 2
        assert all(lock.owned for lock in result.held)
    finally:
        for lock in result.held:
            guard.release_lock(lock)


def test_guard_stops_second_instance_with_exit_code_and_clear_message(capsys: pytest.CaptureFixture) -> None:
    port = free_port()
    keys = make_keys(port)
    first = guard.acquire_instances(keys, grace_sec=0)
    assert len(first.held) == 2
    try:
        with pytest.raises(SystemExit) as stop:
            guard.guard_instances(keys, '127.0.0.1', port, allow=False, grace_sec=0)
        assert stop.value.code == guard.EXIT_ALREADY_RUNNING == 5
        text = capsys.readouterr().err
        assert '二重起動' in text and f'port {port}' in text and 'index 3' in text
        assert 'allow_second_instance' in text      # 外し方を案内する
        assert guard._HELD == []                    # 止めた側は鍵を持ち越さない
    finally:
        for lock in first.held:
            guard.release_lock(lock)


def test_guard_allows_second_instance_with_warning_only(capsys: pytest.CaptureFixture) -> None:
    port = free_port()
    keys = make_keys(port)
    first = guard.acquire_instances(keys, grace_sec=0)
    try:
        guard.guard_instances(keys, '127.0.0.1', port, allow=True, grace_sec=0)   # 止まらない
        text = capsys.readouterr().err
        assert '警告' in text and 'allow_second_instance=true' in text
    finally:
        for lock in first.held + guard._HELD:
            guard.release_lock(lock)


def test_guard_without_conflict_holds_keys_and_is_reentrant(capsys: pytest.CaptureFixture) -> None:
    port = free_port()
    keys = make_keys(port)
    guard.guard_instances(keys, '127.0.0.1', port, allow=False, grace_sec=0)
    try:
        assert capsys.readouterr().err == ''
        assert {lock.key.kind for lock in guard._HELD} == {'port', 'camera'}
        # 同じ process が再び呼んでも自分の鍵と衝突しない (1 process に launcher は 1 つ)
        guard.guard_instances(keys, '127.0.0.1', port, allow=False, grace_sec=0)
        assert len(guard._HELD) == 2
    finally:
        for lock in guard._HELD:
            guard.release_lock(lock)


def test_bind_error_of_free_port_is_none_and_used_port_is_error() -> None:
    port = free_port()
    assert guard.bind_error_of('127.0.0.1', port) is None
    server = socket.socket()
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server.bind(('127.0.0.1', port))
        server.listen()
        assert isinstance(guard.bind_error_of('127.0.0.1', port), OSError)
    finally:
        server.close()
    assert guard.bind_error_of('127.0.0.1', port) is None      # 閉じれば再び空き (probe 自身は port を残さない)


@pytest.mark.skipif(not hasattr(socket, 'SO_EXCLUSIVEADDRUSE'), reason='Windows 専用: 既定 bind (SO_REUSEADDR) の旧版が握る場合')
def test_bind_error_of_sees_a_holder_that_uses_default_reuse_bind() -> None:
    """鍵を持たない旧版 (既定の bind) が port を握っていても、排他の試行は失敗して検出できる。"""
    port = free_port()
    old = socket.socket()
    try:
        old.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        old.bind(('127.0.0.1', port))
        old.listen()
        assert isinstance(guard.bind_error_of('127.0.0.1', port), OSError)
    finally:
        old.close()


def test_guard_stops_when_port_is_held_by_a_process_without_key(capsys: pytest.CaptureFixture) -> None:
    """鍵を持たない相手 (旧版・孤児 process・別アプリ) が port を握っていたら、認識を始める前に止める。"""
    holder = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    holder.bind(('127.0.0.1', 0))
    holder.listen()
    port = holder.getsockname()[1]
    try:
        with pytest.raises(SystemExit) as stop:
            guard.guard_instances(make_keys(port), '127.0.0.1', port, allow=False, grace_sec=0)
        assert stop.value.code == guard.EXIT_ALREADY_RUNNING
        text = capsys.readouterr().err
        assert f'127.0.0.1:{port}' in text and '"port"' in text      # port を変える案内
        assert guard._HELD == []
        guard.guard_instances(make_keys(port), '127.0.0.1', port, allow=True, grace_sec=0)   # 許可なら警告だけ
        assert '警告' in capsys.readouterr().err
    finally:
        holder.close()
        for lock in guard._HELD:
            guard.release_lock(lock)


def test_unusable_os_lock_is_reported_but_does_not_block_startup(monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture) -> None:
    """検出の仕組み自体が OS に拒否されたら、黙らず警告して続行する (検出できないことを名前つきで残す)。"""
    def broken(key: guard.InstanceKey) -> guard.InstanceLock:
        raise PermissionError('アクセスが拒否されました')
    monkeypatch.setattr(guard, 'acquire_lock', broken)
    port = free_port()
    guard.guard_instances(make_keys(port), '127.0.0.1', port, allow=False, grace_sec=0)
    text = capsys.readouterr().err
    assert '検出ができませんでした' in text and '入力カメラ index 3' in text and 'アクセスが拒否' in text


def test_env_policy_default_is_guard_on(monkeypatch: pytest.MonkeyPatch) -> None:
    assert guard.ALLOW_SECOND_INSTANCE_DEFAULT is False
    assert guard.allow_second_instance_from_env() is False and guard.exclusive_bind_default() is True
    monkeypatch.setenv(guard.ALLOW_SECOND_INSTANCE_ENV, '1')
    assert guard.allow_second_instance_from_env() is True and guard.exclusive_bind_default() is False
    monkeypatch.setenv(guard.ALLOW_SECOND_INSTANCE_ENV, '0')
    assert guard.exclusive_bind_default() is True


def test_bind_failure_exit_names_endpoint_and_returns_exit_code(capsys: pytest.CaptureFixture) -> None:
    stop = guard.bind_failure_exit('127.0.0.1', 8765, OSError(10048, 'in use'))
    assert isinstance(stop, SystemExit) and stop.code == guard.EXIT_ALREADY_RUNNING
    assert '127.0.0.1:8765' in capsys.readouterr().err
