"""PuyoLive の二重起動の検出と、HTTP 待受の bind 失敗時の停止。

2026-10-24 の OBS 試験で PuyoLive が 2 つ同時に動き (Start_Auto_Sequence と Manual_NEW24A)、2 つ目は
HTTP 送信 0 件のまま 2 時間認識を続けて CPU を取り合った。Windows での実測 (単体・数秒):
  - ThreadingHTTPServer は既定で SO_REUSEADDR を立てるため、同じ port への 2 つ目の bind が黙って成功する。
    接続は先に bind した側にだけ届く (2 つ目は誰にも配信できない)。
  - SO_EXCLUSIVEADDRUSE を立てた側が先にいれば、2 つ目の bind は WinError 10048 で失敗する。
    既定の bind が後から来ても WinError 10013 で失敗する。
このため検出は 3 段で行う (どれか 1 つが抜けても残りで止まる)。
  1. 名前付きミューテックス (同じ port・同じカメラ index)。起動の最初に取る。process 終了で OS が解放する。
  2. port の排他 bind の試行 (他の PuyoLive・孤児 process・別アプリが握っていないか)。
  3. 評価 process が実際に bind する時の排他 (SO_EXCLUSIVEADDRUSE)。認識を始める前に行う。
利用者が意図して 2 つ動かすときは設定 allow_second_instance=true (警告だけ出して続行し、bind も従来のまま)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Callable, IO

from src.production_config import ALLOW_SECOND_INSTANCE_DEFAULT

# 利用者向けの起動停止の終了コード。launcher の既存値 (0 正常 / 2 設定 / 3 配布物 / 4 実行環境) の続き。
EXIT_ALREADY_RUNNING = 5
# 評価 process (spawn 子) へ「二重起動を許可」を伝える環境変数。launcher が設定値から書く。
ALLOW_SECOND_INSTANCE_ENV = 'PUYO_ALLOW_SECOND_INSTANCE'
ERROR_ALREADY_EXISTS = 183         # Win32: 同名のミューテックスが既にある
MUTEX_NAME_PREFIX = 'Local\\PuyoLive-'  # Local = 同じログオンセッション内 (特権不要)
LOCK_DIR_NAME = 'puyo_live_locks'  # Windows 以外の代替 (flock) の置き場
# 前の PuyoLive が終了処理中でまだ鍵を持っている場合 (再起動の直後) を二重起動と誤認しないための猶予。
# auto_sequence は前の区間の process 木の消滅と port の解放を待ってから次を起動するので通常は 0 秒で通る。
ACQUIRE_GRACE_SEC = 3.0
ACQUIRE_POLL_SEC = 0.25
KIND_PORT, KIND_CAMERA = 'port', 'camera'
SOURCE_DSHOW = 'dshow'


@dataclass(frozen=True)
class InstanceKey:
    """二重起動を判定する資源 1 つ (port か入力カメラ)。"""
    kind: str
    name: str       # ミューテックス名の後半 (英数字のみ)
    label: str      # 利用者向けの表記

    @property
    def mutex_name(self) -> str:
        return MUTEX_NAME_PREFIX + self.name


@dataclass
class InstanceLock:
    """取得した鍵。owned=False は、他の process が先に持っていたことを表す。"""
    key: InstanceKey
    owned: bool
    handle: Any = None


@dataclass
class InstanceCheck:
    held: list[InstanceLock] = field(default_factory=list)
    conflicts: list[InstanceKey] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)   # 検出できなかった資源 (OS のエラー。起動は止めない)


# 取得した鍵は process の寿命だけ保持する (GC で解放されると二重起動を見逃す)。
_HELD: list[InstanceLock] = []


def instance_keys(source: str, host: str, port: int, device_index: int) -> list[InstanceKey]:
    """この起動が占有する資源。動画入力 (開発用) は port だけ。"""
    safe_host = ''.join(ch if ch.isalnum() else '_' for ch in host)
    keys = [InstanceKey(KIND_PORT, f'port-{safe_host}-{port}', f'port {port}')]
    if source == SOURCE_DSHOW:
        keys.append(InstanceKey(KIND_CAMERA, f'camera-{device_index}', f'入力カメラ index {device_index}'))
    return keys


def held_names() -> list[str]:
    """この process が実際に取っている鍵の名前 (起動口の launcher.prepare が働いた process だけが持つ)。
    live_trial_flags.json の instance_guard.held へ残し、検出が効いたことを受領票で確かめる。"""
    return [lock.key.name for lock in _HELD]


def allow_second_instance_from_env() -> bool:
    """子 process が継いだ環境変数。未設定は既定 (二重起動を許可しない)。"""
    return os.environ.get(ALLOW_SECOND_INSTANCE_ENV, str(int(ALLOW_SECOND_INSTANCE_DEFAULT))) == '1'


def exclusive_bind_default() -> bool:
    """HTTP 待受を排他にするか。二重起動を許可しているときだけ従来 (SO_REUSEADDR) のまま。"""
    return not allow_second_instance_from_env()


# ---------------------------------------------------------------- 鍵の取得 (OS 別)

def _kernel32() -> Any:
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _acquire_windows(key: InstanceKey) -> InstanceLock:
    import ctypes
    kernel = _kernel32()
    ctypes.set_last_error(0)
    handle = kernel.CreateMutexW(None, False, key.mutex_name)
    error = ctypes.get_last_error()
    if not handle:
        raise ctypes.WinError(error)
    return InstanceLock(key, owned=error != ERROR_ALREADY_EXISTS, handle=handle)


def _acquire_flock(key: InstanceKey) -> InstanceLock:
    """Windows 以外 (開発・WSL) の代替。flock は開いたファイル記述子ごとなので同じ process 内の 2 回目も衝突する。"""
    import fcntl
    directory = Path(tempfile.gettempdir()) / LOCK_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / f'{key.name}.lock').open('a+b')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        return InstanceLock(key, owned=False)
    return InstanceLock(key, owned=True, handle=stream)


def acquire_lock(key: InstanceKey) -> InstanceLock:
    """鍵を 1 回取る。他が持っていれば owned=False。OS が鍵を作れない場合は OSError。"""
    return _acquire_windows(key) if sys.platform == 'win32' else _acquire_flock(key)


def release_lock(lock: InstanceLock) -> None:
    """鍵を手放す。process 終了時は OS が解放するので、主に試験と競合時の後始末で使う。"""
    handle, lock.handle = lock.handle, None
    if handle is None:
        return
    if sys.platform == 'win32':
        _kernel32().CloseHandle(handle)
    else:
        handle.close()


def acquire_instances(keys: list[InstanceKey], grace_sec: float | None = None,
                      clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep) -> InstanceCheck:
    """全ての鍵を取る。他が持っている鍵は grace_sec まで取り直し、それでも持たれていれば conflicts に入れる。
    この process がすでに持っている鍵は再取得しない (1 process に launcher は 1 つ。再入で自分と衝突しない)。"""
    result = InstanceCheck()
    mine = {lock.key.name for lock in _HELD}
    grace = ACQUIRE_GRACE_SEC if grace_sec is None else grace_sec
    for key in keys:
        if key.name in mine:
            continue
        deadline = clock() + grace
        while True:
            try:
                lock = acquire_lock(key)
            except OSError as error:
                result.errors.append(f'{key.label}: {error}')
                break
            if lock.owned:
                result.held.append(lock)
                break
            release_lock(lock)
            if clock() >= deadline:
                result.conflicts.append(key)
                break
            sleep(ACQUIRE_POLL_SEC)
    return result


# ---------------------------------------------------------------- port の排他 bind

def bind_error_of(host: str, port: int) -> OSError | None:
    """port を排他で bind できるか試して、すぐ閉じる。bind できなければその OSError。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # TIME_WAIT を使用中と誤認しない
        probe.bind((host, port))
    except OSError as error:
        return error
    finally:
        probe.close()
    return None


# ---------------------------------------------------------------- 表示

def conflict_message(conflicts: list[InstanceKey]) -> str:
    labels = '、'.join(key.label for key in conflicts)
    return (f'[二重起動] PuyoLive はすでに起動しています (同じ {labels})。\n'
            '  先に動いている PuyoLive を Ctrl+C で止めてから、もう一度起動してください。\n'
            '  2 つ同時に動かすと認識の CPU を取り合い、2 つ目は配信もできません。\n'
            '  意図して 2 つ動かす場合だけ、設定ファイルに "allow_second_instance": true を書いてください。')


def allowed_message(conflicts: list[InstanceKey]) -> str:
    labels = '、'.join(key.label for key in conflicts)
    return (f'[警告] 同じ {labels} の PuyoLive がすでに動いています。'
            'allow_second_instance=true のため続行します (CPU を取り合い、同じ port なら先に起動した側だけが配信します)。')


def bind_message(host: str, port: int, error: OSError) -> str:
    return (f'[二重起動] {host}:{port} を使えません ({error})。\n'
            '  同じ port を別の PuyoLive や他のアプリが使っているか、Windows の予約範囲に入っています。\n'
            '  動いている PuyoLive を止めるか、設定ファイルの "port" を変えてください。')


def bind_failure_exit(host: str, port: int, error: OSError,
                      err: IO[str] | None = None) -> SystemExit:
    """bind 失敗を利用者に見せ、終了コード付きの SystemExit を返す (呼び出し側が raise する)。"""
    print(bind_message(host, port, error), file=err or sys.stderr, flush=True)
    return SystemExit(EXIT_ALREADY_RUNNING)


# ---------------------------------------------------------------- launcher からの入口

def guard_instances(keys: list[InstanceKey], host: str, port: int, allow: bool,
                    grace_sec: float | None = None, err: IO[str] | None = None) -> None:
    """起動の最初に呼ぶ。二重起動なら案内を出して SystemExit(EXIT_ALREADY_RUNNING)。
    allow=True のときは警告だけで続行する (鍵は保持し、bind の排他も外れる)。"""
    err = err or sys.stderr
    result = acquire_instances(keys, grace_sec)
    for note in result.errors:
        print(f'[警告] 二重起動の検出ができませんでした ({note})。検出なしで続行します。', file=err, flush=True)
    # 鍵が無くても port を握っている相手 (鍵を持たない旧版・孤児 process・別アプリ) は bind の試行で見つける
    bind_error = bind_error_of(host, port) if not any(k.kind == KIND_PORT for k in result.conflicts) else None
    problems = result.conflicts + ([keys[0]] if bind_error is not None else [])
    if not problems:
        _HELD.extend(result.held)
        return
    if allow:
        _HELD.extend(result.held)
        print(allowed_message(problems), file=err, flush=True)
        return
    for lock in result.held:
        release_lock(lock)
    if result.conflicts:
        print(conflict_message(result.conflicts), file=err, flush=True)
        raise SystemExit(EXIT_ALREADY_RUNNING)
    raise bind_failure_exit(host, port, bind_error, err)
