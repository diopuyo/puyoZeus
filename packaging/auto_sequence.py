"""23y の OBS 試験の進行役: OFF → X → ALL → ALL → X → OFF の 6 区間を自動で回す (入口に置く。標準ライブラリだけ)。

アプリ本体のコードは変えない。各区間は、手で起動口を押したときと同じ入口 (start_obs.ps1 → start_obs_trial.py → launcher)
を、別のコンソールで起動して回す。切り替えは user が黒い画面で Ctrl+C を押すのと同じ経路で止める:
  小さな補助 process (--send-ctrl-c) が FreeConsole → AttachConsole(区間の pid) → SetConsoleCtrlHandler(NULL, TRUE) →
  GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0) を行う。区間側は live_lifetime の KeyboardInterrupt → 子へ SIGINT →
  記録の書き出し、の通常の終了を通る。60 秒で終わらなければ強制終了して manifest に残す。
区間の長さ: 「表示が出てから」15 分 (http://127.0.0.1:8765/latest が最初に正しく返った時刻から数える)。
  15 分を過ぎたら、その試合が終わったところで切り替える。試合の終わりの信号は、配信がすでに出している
  /latest の display.match_phase (playing → between_matches) と display.message の「決着」。最大 3 分待ち、
  取れなければ 15 分 + 3 分で切り替える。区間の間に一度も playing を見なかった (信号が無い) ときは 15 分ちょうど。
安全: 次の区間の前に、前の区間の process が全部消えていること・port が空いていることを確かめる。起動の門 (空きメモリ 8.5GiB・
  C: 空き 10GiB) で足りなければ 30 秒おきに最大 5 分まで待つ。区間が落ちたら同じ設定で 1 回だけ起動し直し (残りの時間を回す)、
  2 回目も落ちたら次の区間へ進む。進行役の窓で Ctrl+C なら、今の区間を止めて全体を終える。
記録: 区間ごとの開始・表示開始・停止要求・終了の時刻 (壁時計の ms)・終了コード・output のフォルダ名を manifest.json に残す。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.request import urlopen

DEFAULT_SEQUENCE = ('OFF', 'X', 'ALL', 'ALL', 'X', 'OFF')
ARM_DIRS = {'OFF': '.', 'X': 'x_on', 'ALL': 'all_on'}
SEGMENT_SEC = 15*60.            # 表示が出てからの区間の長さ
GRACE_SEC = 3*60.               # 試合の終わりを待つ最大の長さ
MATCH_END_DEBOUNCE_SEC = 1.5    # 試合の外の表示が、この秒数続いたら試合の終わりとみなす (1 回のちらつきでは切り替えない)
POLL_SEC = .5
DISPLAY_START_TIMEOUT_SEC = 600.
STOP_WAIT_SEC = 60.
HELPER_WAIT_SEC = 15.
PORT_FREE_TIMEOUT_SEC = 60.
LEFTOVER_WAIT_SEC = 90.
TRACK_PERIOD_SEC = 5.
MEMORY_MIN_GIB = 8.5            # start_obs.ps1 の起動の門と同じ
DISK_MIN_GIB = 10.              # 同上 (C: の空き)
GATE_RETRY_SEC = 30.
GATE_RETRY_MAX_SEC = 5*60.
MAX_RESTARTS = 1
UI_PERIOD_SEC = 1.
PLAYING, BETWEEN_MATCHES, DECIDED_PREFIX = 'playing', 'between_matches', '決着'
SW_SHOWMINNOACTIVE = 7          # 区間の窓は最小化・非アクティブで開く (対戦中の画面の邪魔をしない)
CTRL_C_EVENT = 0
DETACHED_PROCESS = 0x00000008
STILL_ACTIVE_WAIT = 0x00000102
# 強制終了してよい実行ファイル名 (区間の process が実際に出す名前だけ。小文字で比べる)。これ以外 (obs64.exe・explorer.exe・
# 別アプリなど) は、子孫に見えても止めず、記録 (leftovers_skipped) だけを残す。
KILLABLE_EXES = frozenset(('python.exe', 'pythonw.exe', 'powershell.exe', 'pwsh.exe', 'cmd.exe', 'conhost.exe',
                           'ama_reference.exe', 'ffmpeg.exe', 'nvidia-smi.exe'))
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
TH32CS_SNAPPROCESS = 0x2
INVALID_HANDLE = ctypes.c_void_p(-1).value
LATEST_TIMEOUT_SEC = 1.
# 煙試験 (video モード) が start_obs.ps1 / start_obs_trial.py と同じ環境で起動するための値 (入口の start_obs_trial.py と同じ)。
VIDEO_ENV = dict(PUYO_OBS_CPU_ISOLATION='0', PUYO_OBS_AMA_IDLE='0', PUYO_OBS_FAST_AUDIT='1', PUYO_OBS_EXACT_SEED='1',
                 PUYO_OBS_PREPARE_EXACT='1', PUYO_OBS_DEEP_TIMING='1', PUYO_OBS_STDLIB_CACHE='1',
                 PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')


def kill_report(report: dict[str, dict[int, str]]) -> dict[str, dict[str, str]]:
    """kill_handle の結果 (止めた・止めず記録だけ) を、manifest へ書ける形 (pid を文字列) にする。"""
    return {kind: {str(pid): exe for pid, exe in found.items()} for kind, found in report.items()}


def absolute(path: Path | str) -> Path:
    """絶対パス (マップドライブの文字のまま)。Path.resolve() は D: を UNC (サーバー名つきのパス) へ変えてしまい、start_obs.ps1 の
    Resolve-Path がプロバイダ名つきのパスを返して、起動の記録 (--receipt) が書けず起動が落ちる (D: の入口で動かしたとき)。"""
    return Path(os.path.abspath(path))


# --- 試合の終わりの判定 (純粋) -----------------------------------------------------------------------------------

def snapshot_phase(snapshot: Any) -> str | None:
    """/latest の display.match_phase。形が違えば None (信号が無い)。"""
    display = snapshot.get('display') if isinstance(snapshot, dict) else None
    phase = display.get('match_phase') if isinstance(display, dict) else None
    return phase if isinstance(phase, str) else None


def snapshot_match_over(snapshot: Any) -> bool:
    """試合の外 (次試合待ち) か、決着の表示 (display.message が「決着」で始まる)。"""
    display = snapshot.get('display') if isinstance(snapshot, dict) else None
    message = display.get('message') if isinstance(display, dict) else None
    return snapshot_phase(snapshot) == BETWEEN_MATCHES or (isinstance(message, str) and message.startswith(DECIDED_PREFIX))


@dataclass
class Decision:
    stop: bool
    reason: str | None = None


@dataclass
class SegmentTimer:
    """区間の時計。表示が出てから segment_sec、そのあと試合の終わりを grace_sec まで待つ。時刻は呼び出し側が渡す。

    区間が途中で落ちて起動し直したときも、表示していた時間は足し合わせて残りを回す (begin_attempt)。
    """

    segment_sec: float = SEGMENT_SEC
    grace_sec: float = GRACE_SEC
    debounce_sec: float = MATCH_END_DEBOUNCE_SEC
    carried_sec: float = 0.
    display_start: float | None = None
    saw_playing: bool = False
    over_since: float | None = None

    def begin_attempt(self, now: float) -> None:
        """起動し直す前に、今までの表示の時間を持ち越す。"""
        self.carried_sec = self.elapsed(now)
        self.display_start, self.over_since = None, None

    def elapsed(self, now: float) -> float:
        return self.carried_sec+(0. if self.display_start is None else now-self.display_start)

    def observe(self, now: float, snapshot: Any) -> Decision:
        """1 回の /latest の結果 (取れなければ None) を見て、止めるかを返す。"""
        if snapshot is None:
            return self.decide(now)
        if self.display_start is None:
            self.display_start = now
        if snapshot_match_over(snapshot):
            self.over_since = now if self.over_since is None else self.over_since
        else:
            self.over_since = None
            self.saw_playing = self.saw_playing or snapshot_phase(snapshot) == PLAYING
        return self.decide(now)

    def deadline(self) -> float | None:
        """15 分に達する時刻 (表示が出ていなければ None)。"""
        return None if self.display_start is None else self.display_start+self.segment_sec-self.carried_sec

    def decide(self, now: float) -> Decision:
        deadline = self.deadline()
        if deadline is None or now < deadline:
            return Decision(False)
        if not self.saw_playing:
            return Decision(True, 'fixed_time_no_phase_signal')   # 信号が無い: 15 分ちょうど
        if self.over_since is not None and now-self.over_since >= self.debounce_sec:
            return Decision(True, 'match_end' if self.over_since > deadline else 'between_matches_at_deadline')
        if now >= deadline+self.grace_sec:
            return Decision(True, 'grace_timeout')
        return Decision(False)

    def planned_switch(self, now: float, wall_now: float) -> float:
        """次の切り替えの予定 (壁時計の秒)。試合の終わり待ちを含まない最短の予定。"""
        remaining = max(0., self.segment_sec-self.elapsed(now))
        return wall_now+remaining


# --- manifest -----------------------------------------------------------------------------------------------

def wall_ms() -> int:
    return int(time.time()*1000)


def iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


class Manifest:
    """進行の記録。更新のたびに丸ごと書き直す (途中で止まっても、そこまでの記録が残る)。"""

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        self.path, self.data = path, dict(header, segments=[], events=[])
        self.write()

    def event(self, kind: str, **values: Any) -> None:
        self.data['events'].append(dict(kind=kind, wall_ms=wall_ms(), at=iso_ms(wall_ms()), **values))
        self.write()

    def write(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, self.path)


# --- Windows の process 操作 -----------------------------------------------------------------------------------

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD),
                ('th32DefaultHeapID', ctypes.c_void_p), ('th32ModuleID', wintypes.DWORD),
                ('cntThreads', wintypes.DWORD), ('th32ParentProcessID', wintypes.DWORD),
                ('pcPriClassBase', wintypes.LONG), ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR*260)]


def kernel32() -> Any:
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel


def process_table() -> dict[int, tuple[int, str]]:
    """{pid: (親の pid, exe 名)} の全 process (Toolhelp)。"""
    kernel = kernel32()
    snapshot = kernel.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    table: dict[int, tuple[int, str]] = {}
    if snapshot in (None, INVALID_HANDLE):
        return table
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
    while more:
        table[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
        more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    kernel.CloseHandle(snapshot)
    return table


def born_after(parent: int, child: int, created: Callable[[int], int | None]) -> bool:
    """子の作成時刻が親以降か。どちらかが取れなければ False (確かでないものは子孫に入れない)。"""
    parent_time, child_time = created(parent), created(child)
    return parent_time is not None and child_time is not None and child_time >= parent_time


def descendants(root: int, table: dict[int, tuple[int, str]],
                created: Callable[[int], int | None] | None = None) -> dict[int, str]:
    """root の子孫 (root を含む) の {pid: exe 名}。親が死んだ後に残る孫は、次の走査では辿れない。

    created (pid → 作成時刻) を渡すと、子の作成時刻が親より前のものを外す。親の pid が使い回されると、
    死んだ親を指したままの関係のない process (例: 別アプリ) が子孫に見え、後で止めてしまうため。
    """
    found = {root: table.get(root, (0, '?'))[1]}
    changed = True
    while changed:
        changed = False
        for pid, (parent, exe) in table.items():
            if parent in found and pid not in found and pid != parent and (created is None or born_after(parent, pid, created)):
                found[pid], changed = exe, True
    return found


def kill_plan(root: int, tracked: dict[int, tuple[str, int | None]], table: dict[int, tuple[int, str]],
              created: Callable[[int], int | None], alive: Callable[[int, int | None], bool],
              allowed: frozenset[str] = KILLABLE_EXES) -> tuple[list[tuple[int, str, int]], dict[int, str]]:
    """止める順 [(pid, exe, 作成時刻)] (子から先) と、止めずに記録だけ残す {pid: exe} を返す (純粋)。

    候補 = 記録済みの子孫 (作成時刻つき) + 今の子孫 (子の作成時刻が親以降のものだけ)。そのうち今も生きていて
    (pid の使い回しなら作成時刻が変わるので除かれる)、実行ファイル名が allowed にあるものが止める対象。
    """
    candidates = {pid: (exe, born) for pid, (exe, born) in tracked.items() if born is not None}
    for pid, exe in descendants(root, table, created).items():
        born = created(pid)
        if pid not in candidates and born is not None:
            candidates[pid] = (exe, born)
    live = {pid: item for pid, item in candidates.items() if alive(pid, item[1])}

    def depth(pid: int) -> int:
        steps, parent = 0, table.get(pid, (0, ''))[0]
        while parent in live and parent != pid and steps < len(live):     # 使い回しで輪になっても止まる上限
            steps, parent = steps+1, table.get(parent, (0, ''))[0]
        return steps
    names = {pid: item[0] for pid, item in live.items()}
    order = [(pid, live[pid][0], live[pid][1]) for pid in sorted(live, key=lambda p: (-depth(p), p))
             if live[pid][0].lower() in allowed]
    return order, {pid: exe for pid, exe in names.items() if exe.lower() not in allowed}


def creation_time(pid: int) -> int | None:
    """process の作成時刻 (FILETIME)。pid の使い回しを見分ける。死んでいれば None。"""
    kernel = kernel32()
    handle = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in (created, exited, kernel_time, user_time))):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def is_alive(pid: int, created: int | None) -> bool:
    """pid の process が、記録した作成時刻のまま生きているか。"""
    now = creation_time(pid)
    if now is None or (created is not None and now != created):
        return False
    kernel = kernel32()
    handle = kernel.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == STILL_ACTIVE_WAIT
    finally:
        kernel.CloseHandle(handle)


def send_ctrl_c(pid: int) -> None:
    """補助 process の本体。pid のコンソールへ、手で Ctrl+C を押したのと同じ CTRL_C_EVENT を送る。"""
    kernel = kernel32()
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise OSError(ctypes.get_last_error(), f'AttachConsole({pid}) に失敗')
    kernel.SetConsoleCtrlHandler(None, True)           # この補助 process 自身は Ctrl+C を無視する
    if not kernel.GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0):
        raise OSError(ctypes.get_last_error(), 'GenerateConsoleCtrlEvent に失敗')
    time.sleep(1.)                                        # 配送を待ってから、コンソールを離れる
    kernel.FreeConsole()


# --- 実機の操作 (試験では差し替える) ----------------------------------------------------------------------------

@dataclass
class Handle:
    """起動した区間 1 回分。tracked = 観測した子孫 {pid: (exe, 作成時刻)}。"""

    popen: Any
    arm_dir: Path
    started_wall_ms: int
    before_runs: set[str] = field(default_factory=set)
    tracked: dict[int, tuple[str, int | None]] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)


class Platform:
    """実機の操作。FakePlatform (試験) がこれと同じ口を持つ。"""

    def __init__(self, bundle: Path, entry: Path, runtime_python: str, mode: str = 'obs') -> None:
        self.bundle, self.entry, self.runtime_python, self.mode = bundle, entry, runtime_python, mode

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def free_gib(self) -> tuple[float, float]:
        """(空きメモリ GiB, 入口のあるドライブの空き GiB)。"""
        class Status(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD), ('total', ctypes.c_ulonglong),
                        ('avail', ctypes.c_ulonglong), ('p1', ctypes.c_ulonglong), ('p2', ctypes.c_ulonglong),
                        ('v1', ctypes.c_ulonglong), ('v2', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(status)
        ctypes.WinDLL('kernel32').GlobalMemoryStatusEx(ctypes.byref(status))
        import shutil
        return status.avail/2**30, shutil.disk_usage(self.entry).free/2**30

    def port_free(self, port: int) -> bool:
        with socket.socket() as client:
            client.settimeout(.2)
            return client.connect_ex(('127.0.0.1', port)) != 0

    def fetch_latest(self, port: int) -> Any:
        try:
            with urlopen(f'http://127.0.0.1:{port}/latest', timeout=LATEST_TIMEOUT_SEC) as response:
                value = json.load(response)
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def resolve_camera(self, arm_dir: Path) -> int:
        """機器名から device_index を puyo_live.json へ書く (手で起動口を押したときと同じ)。"""
        script = arm_dir/'resolve_camera.ps1'
        result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)],
                                capture_output=True, cwd=arm_dir)
        return result.returncode

    def command(self, arm_dir: Path) -> list[str]:
        """obs モードの起動の命令: start_obs.ps1 (手で押す起動口と同じ)。"""
        return ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(self.entry/'start_obs.ps1'),
                '-Bundle', str(self.bundle), '-Entry', str(arm_dir)]

    def video_launch(self, arm_dir: Path) -> tuple[list[str], dict[str, str]]:
        """video モード (煙試験用): start_obs.ps1 と start_obs_trial.py がする準備 (起動の記録・走行の記録の置き場・設定の写し)
        を同じ形でして、動画入力を許す配布の起動口 (bootstrap --video) の命令と環境を返す。"""
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S_%f')
        startup, run = arm_dir/'output'/('startup_'+stamp), arm_dir/'output'/stamp
        (startup/'temp').mkdir(parents=True)
        run.mkdir(parents=True)
        values = json.loads((arm_dir/'puyo_live.json').read_text(encoding='utf-8-sig'))
        for key, name in (('b3_model', 'model.joblib'), ('b3_timing_table', 'timing_table.json')):
            values[key] = str(self.bundle/'app/models/b3'/name)
        values['output_dir'] = str(run)
        (run/'puyo_live.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
        runtime = Path(self.runtime_python).parent
        command = [self.runtime_python, '-B', '-u', str(self.bundle/'obs_runtime_bootstrap.py'), '--bundle', str(self.bundle),
                   '--runtime', str(runtime), '--config', str(run/'puyo_live.json'),
                   '--receipt', str(startup/'runtime_receipt.json'), '--video']
        return command, dict(os.environ, TEMP=str(startup/'temp'), TMP=str(startup/'temp'), **VIDEO_ENV)

    def runs_in(self, arm_dir: Path) -> set[str]:
        output = arm_dir/'output'
        return {p.name for p in output.iterdir() if p.is_dir()} if output.is_dir() else set()

    def launch(self, arm_dir: Path) -> Handle:
        """別のコンソール (最小化・非アクティブ) で起動する。"""
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = SW_SHOWMINNOACTIVE
        before = self.runs_in(arm_dir)
        command, environment = (self.command(arm_dir), None) if self.mode == 'obs' else self.video_launch(arm_dir)
        popen = subprocess.Popen(command, cwd=self.bundle, creationflags=subprocess.CREATE_NEW_CONSOLE,
                                 startupinfo=info, env=environment)
        return Handle(popen, arm_dir, wall_ms(), before, command=command)

    def poll(self, handle: Handle) -> int | None:
        return handle.popen.poll()

    def track(self, handle: Handle) -> None:
        """今の子孫を tracked へ足す (作成時刻つき)。"""
        for pid, exe in descendants(handle.popen.pid, process_table(), creation_time).items():
            handle.tracked.setdefault(pid, (exe, creation_time(pid)))

    def request_stop(self, handle: Handle) -> int:
        """補助 process で Ctrl+C を送る。補助の終了コードを返す。"""
        helper = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--send-ctrl-c',
                                   str(handle.popen.pid)], creationflags=DETACHED_PROCESS)
        try:
            return helper.wait(timeout=HELPER_WAIT_SEC)
        except subprocess.TimeoutExpired:
            helper.kill()
            return -1

    def wait_exit(self, handle: Handle, seconds: float) -> int | None:
        try:
            return handle.popen.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            return None

    def kill_handle(self, handle: Handle) -> dict[str, dict[int, str]]:
        """この区間の process だけを、子から先に 1 つずつ強制終了する。taskkill の /T は使わない。

        /T は Windows がその時点の親 pid で子をたどるため、親の pid が使い回されると関係のない process (別アプリ・OBS) まで止める。
        止める対象は、作成時刻の門を通った子孫 + 記録済みの子孫のうち、止める直前に作成時刻が記録のままで生きているものだけ。
        さらに実行ファイル名が KILLABLE_EXES にあるものに限り、それ以外 (obs64.exe など) は止めずに skipped へ記録する。
        """
        self.track(handle)
        order, skipped = kill_plan(handle.popen.pid, handle.tracked, process_table(), creation_time, is_alive)
        killed: dict[int, str] = {}
        for pid, exe, born in order:
            if is_alive(pid, born):
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True)
                killed[pid] = exe
        return dict(killed=killed, skipped=skipped)

    def alive(self, handle: Handle) -> dict[int, str]:
        """tracked のうち、まだ生きている {pid: exe 名}。"""
        return {pid: exe for pid, (exe, created) in handle.tracked.items() if is_alive(pid, created)}

    def wall_ms(self) -> int:
        return wall_ms()


# --- 進行 ----------------------------------------------------------------------------------------------------

class SequenceAborted(Exception):
    """進行役の窓で Ctrl+C が押され、今の区間を止めて全体を終えた。"""


@dataclass
class Settings:
    sequence: tuple[str, ...] = DEFAULT_SEQUENCE
    segment_sec: float = SEGMENT_SEC
    grace_sec: float = GRACE_SEC
    display_timeout_sec: float = DISPLAY_START_TIMEOUT_SEC
    stop_wait_sec: float = STOP_WAIT_SEC


def arm_port(arm_dir: Path) -> int:
    """区間の設定 (puyo_live.json) の port。読めなければ 8765。"""
    try:
        return int(json.loads((arm_dir/'puyo_live.json').read_text(encoding='utf-8-sig')).get('port', 8765))
    except (OSError, ValueError):
        return 8765


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def shutdown_statuses(run: Path) -> dict[str, Any]:
    """手で止めたときの終了検査 (煙試験 shutdown_checks と同じ原票): 強制終了が無く、投入と処理の件数が合うか。"""
    statuses: dict[str, Any] = {}
    for name in ('b3_async_status.json', 'v49f_b3_v6/b3_async_status.json'):
        value = read_json(run/name) or {}
        statuses[name] = bool(value) and not value.get('forced_termination') and \
            (value.get('finished') or {}).get('inputs') == value.get('submitted')
    value = read_json(run/'v49f_display_status.json') or {}
    statuses['v49f_display_status.json'] = bool(value) and not value.get('failure') and not value.get('pump_alive') \
        and value.get('submitted') == value.get('received')
    for name in ('v49f_async_status.json', 'ama_async_status.json'):
        value = read_json(run/name) or {}
        statuses[name] = bool(value) and value.get('exit_code') == 0 and not value.get('forced_termination') \
            and not value.get('pump_alive')
    return dict(passed=all(statuses.values()), statuses=statuses)


class Sequencer:
    """6 区間を順に回す。実機の操作は platform が持つ (試験では FakePlatform)。"""

    def __init__(self, platform: Any, entry: Path, manifest: Manifest, settings: Settings,
                 out: Callable[[str], None] = print) -> None:
        self.platform, self.entry, self.manifest, self.settings, self.out = platform, entry, manifest, settings, out
        self.last_ui = float('-inf')

    def arm_dir(self, arm: str) -> Path:
        return absolute(self.entry/ARM_DIRS[arm])

    def run(self) -> int:
        """全区間を回す。戻り値は終了コード (0 = 最後まで回した、1 = 落ちた区間がある、2 = Ctrl+C で止めた)。"""
        failed = False
        try:
            for index, arm in enumerate(self.settings.sequence, 1):
                record = self.run_segment(index, arm)
                failed = failed or record['status'] != 'finished'
        except (SequenceAborted, KeyboardInterrupt) as error:
            self.manifest.data['status'] = 'aborted_by_user'
            self.manifest.event('aborted_by_user', during_watch=isinstance(error, SequenceAborted))
            self.out('\n進行役を止めました (今の区間は止めて記録を残しました)')
            return 2
        self.manifest.data['status'] = 'finished_with_failures' if failed else 'finished'
        self.manifest.event('finished', failed=failed)
        self.out('\n===== 終了: 6 区間が終わりました。OBS の録画と仮想カメラを止めてください =====')
        return 1 if failed else 0

    def run_segment(self, index: int, arm: str) -> dict[str, Any]:
        directory = self.arm_dir(arm)
        record: dict[str, Any] = dict(index=index, arm=arm, arm_dir=str(directory), port=arm_port(directory),
                                      status='running', attempts=[])
        self.manifest.data['segments'].append(record)
        self.manifest.event('segment_begin', index=index, arm=arm)
        started = datetime.fromtimestamp(self.platform.wall_ms()/1000).strftime('%H:%M:%S')
        self.out(f'\n--- 区間 {index}/{len(self.settings.sequence)} [{arm}] を起動します ({started})')
        timer = SegmentTimer(self.settings.segment_sec, self.settings.grace_sec)
        for attempt in range(1, MAX_RESTARTS+2):
            result = self.attempt(record, timer, attempt)
            record['status'] = result
            if result == 'finished':
                break
            self.manifest.event('segment_attempt_failed', index=index, arm=arm, attempt=attempt, result=result)
            if result == 'gate_timeout' or timer.elapsed(self.platform.monotonic()) >= self.settings.segment_sec:
                break
            timer.begin_attempt(self.platform.monotonic())
        record['displayed_sec'] = round(timer.elapsed(self.platform.monotonic()), 1)
        self.manifest.event('segment_end', index=index, arm=arm, status=record['status'])
        return record

    def wait_gate(self, attempt: dict[str, Any]) -> bool:
        """起動の門 (空きメモリ・C: の空き)。足りなければ 30 秒おきに最大 5 分まで待つ。"""
        waited = 0.
        while True:
            memory, disk = self.platform.free_gib()
            if memory >= MEMORY_MIN_GIB and disk >= DISK_MIN_GIB:
                attempt['gate_waited_sec'] = waited
                return True
            if waited >= GATE_RETRY_MAX_SEC:
                attempt.update(gate_waited_sec=waited, gate_memory_gib=round(memory, 2), gate_disk_gib=round(disk, 1))
                return False
            self.out(f'起動の門: 空きメモリ {memory:.1f}GiB / 空き {disk:.0f}GiB '
                     f'(必要 {MEMORY_MIN_GIB}/{DISK_MIN_GIB}) 30 秒待ちます')
            self.platform.sleep(GATE_RETRY_SEC)
            waited += GATE_RETRY_SEC

    def wait_port_free(self, port: int) -> bool:
        waited = 0.
        while not self.platform.port_free(port):
            if waited >= PORT_FREE_TIMEOUT_SEC:
                return False
            self.platform.sleep(1.)
            waited += 1.
        return True

    def attempt(self, record: dict[str, Any], timer: SegmentTimer, number: int) -> str:
        """区間 1 回分の起動から終了まで。finished / crashed / startup_failed / gate_timeout / port_busy を返す。"""
        directory, port = Path(record['arm_dir']), record['port']
        info: dict[str, Any] = dict(attempt=number)
        record['attempts'].append(info)
        if not self.wait_gate(info):
            return 'gate_timeout'
        if not self.wait_port_free(port):
            info['port_busy'] = True
            return 'port_busy'
        if getattr(self.platform, 'mode', 'obs') == 'obs' and self.platform.resolve_camera(directory) != 0:
            info['camera_unresolved'] = True
            return 'startup_failed'
        handle = self.platform.launch(directory)
        info.update(started_wall_ms=handle.started_wall_ms, started_at=iso_ms(handle.started_wall_ms),
                    command=handle.command, pid=handle.popen.pid)
        self.manifest.event('segment_started', index=record['index'], arm=record['arm'], attempt=number)
        try:
            result = self.watch(record, info, handle, timer)
        except KeyboardInterrupt:
            self.stop(info, handle, 'user_abort')
            self.finish(info, handle)
            raise SequenceAborted() from None
        self.finish(info, handle)
        return 'finished' if result == 'stopped' else result

    def watch(self, record: dict[str, Any], info: dict[str, Any], handle: Handle, timer: SegmentTimer) -> str:
        """区間を見張る。stopped = 時計に従って止めた、crashed / startup_failed = 止める前に落ちた。"""
        launched, tracked_at = self.platform.monotonic(), float('-inf')
        while True:
            now = self.platform.monotonic()
            code = self.platform.poll(handle)
            if code is not None:
                info.update(exit_code=code, exited_wall_ms=self.platform.wall_ms(), exited_before_stop=True)
                return 'crashed' if timer.display_start is not None else 'startup_failed'
            if now-tracked_at >= TRACK_PERIOD_SEC:
                self.platform.track(handle)
                tracked_at = now
            decision = self.observe(record, info, timer, now)
            if decision.stop:
                self.stop(info, handle, decision.reason or 'timer')
                return 'stopped'
            if timer.display_start is None and now-launched > self.settings.display_timeout_sec:
                info['display_timeout'] = True
                info['forced_kill_report'] = kill_report(self.platform.kill_handle(handle))
                return 'startup_failed'
            self.show(record, timer, now)
            self.platform.sleep(POLL_SEC)

    def observe(self, record: dict[str, Any], info: dict[str, Any], timer: SegmentTimer, now: float) -> Decision:
        """/latest を 1 回見て時計へ渡す。表示が最初に出た瞬間を記録する。"""
        started = timer.display_start is not None
        decision = timer.observe(now, self.platform.fetch_latest(record['port']))
        if not started and timer.display_start is not None:
            info.update(display_start_wall_ms=self.platform.wall_ms())
            info['display_start_at'] = iso_ms(info['display_start_wall_ms'])
            self.manifest.event('display_started', index=record['index'], arm=record['arm'])
        return decision

    def show(self, record: dict[str, Any], timer: SegmentTimer, now: float) -> None:
        """進行役の画面: 今の区間・経過・次の切り替えの予定。"""
        if now-self.last_ui < UI_PERIOD_SEC:
            return
        self.last_ui = now
        total, elapsed, limit = len(self.settings.sequence), int(timer.elapsed(now)), int(self.settings.segment_sec)
        plan = datetime.fromtimestamp(timer.planned_switch(now, self.platform.wall_ms()/1000.)).strftime('%H:%M:%S')
        if timer.display_start is None:
            state = '起動中'
        else:
            state = '試合の終わり待ち' if elapsed >= limit else '計測中'
        self.out(f"\r区間 {record['index']}/{total} [{record['arm']}]  経過 {elapsed//60:02d}:{elapsed % 60:02d} / "
                 f"{limit//60:02d}:{limit % 60:02d}  次の切り替え予定 {plan}  ({state})   ")

    def stop(self, info: dict[str, Any], handle: Handle, reason: str) -> None:
        """Ctrl+C と同じ経路で止める。60 秒で終わらなければ強制終了して記録に残す。"""
        info.update(stop_reason=reason, stop_requested_wall_ms=self.platform.wall_ms())
        info['stop_requested_at'] = iso_ms(info['stop_requested_wall_ms'])
        self.manifest.event('stop_requested', reason=reason, pid=handle.popen.pid)
        self.out(f'\n    切り替え: {reason} (Ctrl+C と同じ経路で止めます)')
        self.platform.track(handle)
        info['helper_exit_code'] = self.platform.request_stop(handle)
        code = self.platform.wait_exit(handle, self.settings.stop_wait_sec)
        info['forced_kill'] = code is None
        if code is None:
            info['forced_kill_report'] = kill_report(self.platform.kill_handle(handle))
            code = self.platform.wait_exit(handle, HELPER_WAIT_SEC)
        info.update(exit_code=code, exited_wall_ms=self.platform.wall_ms())

    def finish(self, info: dict[str, Any], handle: Handle) -> None:
        """終了後の確認: 子孫が全部消えたか (残れば止める)・port が空いたか・出力のフォルダ名・終了時の記録。"""
        directory, waited = handle.arm_dir, 0.
        info.setdefault('exited_wall_ms', self.platform.wall_ms())
        while self.platform.alive(handle) and waited < LEFTOVER_WAIT_SEC:
            self.platform.sleep(1.)
            waited += 1.
        leftovers = self.platform.alive(handle)
        report = kill_report(self.platform.kill_handle(handle)) if leftovers else dict(killed={}, skipped={})
        info.update(tracked_processes=len(handle.tracked), tracked_exes=sorted({exe for exe, _ in handle.tracked.values()}),
                    leftovers_killed=report['killed'], leftovers_skipped=report['skipped'])
        info['port_free_after'] = self.platform.port_free(arm_port(directory))
        new = sorted(self.platform.runs_in(directory)-handle.before_runs)
        runs = [name for name in new if not name.startswith('startup_')]
        startup = [name for name in new if name.startswith('startup_')]
        info.update(output_dir=runs[-1] if runs else None, startup_dir=startup[-1] if startup else None)
        if runs:
            info['shutdown'] = shutdown_statuses(directory/'output'/runs[-1])
        if startup:
            failure = read_json(directory/'output'/startup[-1]/'runtime_failure.json')
            info['runtime_failure'] = failure.get('exception') if isinstance(failure, dict) else None
        self.manifest.event('segment_finished', output_dir=info['output_dir'], exit_code=info.get('exit_code'))


# --- 起動口 ----------------------------------------------------------------------------------------------------

def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--sequence', default=','.join(DEFAULT_SEQUENCE))
    parser.add_argument('--segment-sec', type=float, default=SEGMENT_SEC)
    parser.add_argument('--grace-sec', type=float, default=GRACE_SEC)
    parser.add_argument('--stop-wait-sec', type=float, default=STOP_WAIT_SEC)
    parser.add_argument('--mode', choices=('obs', 'video'), default='obs',
                        help='video は煙試験用 (動画入力の config を直接起動)')
    parser.add_argument('--runtime-python', default=_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python/python.exe'))
    parser.add_argument('--send-ctrl-c', type=int, help='内部用: この pid のコンソールへ Ctrl+C を送る補助')
    return parser.parse_args(argv)


def console_out(text: str) -> None:
    """進行役の画面への出力。キャリッジリターンで始まる 1 行は、その場の書き換え (改行しない)。"""
    print(text, end='' if text.startswith('\r') else '\n', flush=True)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.send_ctrl_c is not None:
        send_ctrl_c(args.send_ctrl_c)
        return 0
    if args.bundle is None or args.entry is None:
        raise SystemExit('--bundle と --entry が必要です')
    sequence = tuple(arm.strip().upper() for arm in args.sequence.split(','))
    unknown = [arm for arm in sequence if arm not in ARM_DIRS]
    if unknown:
        raise SystemExit(f'知らない腕: {unknown} (OFF / X / ALL)')
    # 親から継いだ「Ctrl+C を無視」を解く。残っていると、起動した区間も Ctrl+C を無視して止まらない。
    kernel32().SetConsoleCtrlHandler(None, False)
    folder = absolute(args.entry)/'auto_sequence'/datetime.now().strftime('%Y%m%dT%H%M%S')
    folder.mkdir(parents=True)
    settings = Settings(sequence, args.segment_sec, args.grace_sec, DISPLAY_START_TIMEOUT_SEC, args.stop_wait_sec)
    manifest = Manifest(folder/'manifest.json', dict(
        kind='obs_23y_auto_sequence', started_wall_ms=wall_ms(), started_at=iso_ms(wall_ms()), bundle=str(args.bundle),
        entry=str(absolute(args.entry)), mode=args.mode, sequence=list(sequence), segment_sec=args.segment_sec,
        grace_sec=args.grace_sec, signal='/latest display.match_phase (playing -> between_matches) / display.message 決着',
        status='running'))
    arrow = ' -> '.join(sequence)
    print('USER ONLY - DO NOT DISTRIBUTE')
    print(f'進行役: {arrow} (表示が出てから {int(args.segment_sec)//60} 分ずつ)')
    print(f'記録: {folder}')
    print('この窓で Ctrl+C を押すと、今の区間を止めて全体を終えます。', flush=True)
    platform = Platform(args.bundle, absolute(args.entry), args.runtime_python, args.mode)
    return Sequencer(platform, args.entry, manifest, settings, out=console_out).run()


if __name__ == '__main__':
    sys.exit(main())
