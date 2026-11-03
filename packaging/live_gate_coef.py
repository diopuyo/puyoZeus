"""通常は説明だけ。実行語と確認語の両方が揃ってから全長ABBAを開始する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_gate_power as power
from live_gate_win import create_job, kernel, masks, topology, NORMAL_PRIORITY, singleton, process_birth

ROOT = Path('D:/puyo_analyzer/packaging/live_gate_coef_20261022')
BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261021c_stage/PuyoLive')
VIDEO = Path('C:/puyo1_tools/soak/media/video_d04.mp4')
RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))
DLL = Path(_public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'))
CONFIG_SHA = '1eec38bf16c3effe20cf69b1cdd266334522fc74495b472c3942d93f520fe7f1'
MANIFEST_SHA = 'c24df92fb0f69b7f825dd74523bd4fe0410120a66bfad895b879dd5a0df5c737'
VIDEO_SHA = 'db5764831bb023fe090fda4eaed2fbf9e0aaade85efbd27256c13af6539d555a'
RUN_WORD = 'RUN-ABBA-21C'
CONFIRM = 'PC1-IDLE-FULL-ABBA'
COOLDOWN_SEC = 60
MIN_D_GIB = 30
ORDER = ('A1_normal', 'B1_low', 'B2_low', 'A2_normal')


def digest(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def metadata() -> dict:
    """動画は再生せず、コンテナの長さだけ確認する。"""
    os.environ['OPENCV_FFMPEG_DLL_DIR'] = str(DLL)
    import cv2
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or count <= 0 or fps <= 0:
            raise RuntimeError('入力動画の全長を確認できない')
        return dict(frames=count, fps=fps, seconds=count/fps, run_seconds=math.ceil(count/fps))
    finally:
        cap.release()


def preflight() -> dict:
    for path in (VIDEO, BUNDLE/'puyo_live.json', BUNDLE/'app/MANIFEST.json', RUNTIME/'python.exe', DLL):
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(BUNDLE/'puyo_live.json') != CONFIG_SHA:
        raise RuntimeError('21cの配布設定SHAが不一致')
    if digest(BUNDLE/'app/MANIFEST.json') != MANIFEST_SHA:
        raise RuntimeError('21cの配布コード台帳SHAが不一致')
    rows = topology()
    normal, low = masks(rows)
    return dict(topology=rows, normal_mask=normal, low_mask=low, video=metadata(),
        original_plan=power.active(), config_sha256=CONFIG_SHA, app_manifest_sha256=MANIFEST_SHA,
        bundle=str(BUNDLE), video_path=str(VIDEO), profile='four_P_cores_all_siblings_boost_off',
        parent_priority='Normal', synthetic_load=False)


def pending_recovery() -> None:
    pointer = ROOT/'pending.json'
    if not pointer.exists():
        return
    state = Path(json.loads(pointer.read_text(encoding='utf-8'))['state'])
    value = json.loads(state.read_text(encoding='utf-8'))
    if value.get('restored'):
        return
    # 同時起動時に、走行中の電源を戻さない。
    birth = process_birth(value['controller_pid'])
    if birth is not None and birth == value['controller_birth']:
        raise RuntimeError('前回controllerが生存。二重起動しない')
    power.restore(state)


def check_idle(out: Path) -> None:
    from smoke_live_obs_20261020 import memory_guard
    memory_guard()
    if shutil.disk_usage(ROOT).free/1024**3 < MIN_D_GIB:
        raise RuntimeError('D:の空き30GiB未満')
    subprocess.run(['powershell', '-NoProfile', '-File', str(HERE/'wait_live_obs_idle.ps1'),
                    '-Output', str(out/'idle.jsonl'), '-ExcludePid', str(os.getpid()), '-SingleCheck'], check=True)
    last = json.loads((out/'idle.jsonl').read_text(encoding='utf-8-sig').splitlines()[-1])
    if last['busy']:
        raise RuntimeError('PC1は使用中。負荷を止めず、測定を開始しない')
    # AC接続を判別できない場合も、勝手な条件で測らない。
    import ctypes as ct
    status = (ct.c_byte * 12)()
    if not ct.windll.kernel32.GetSystemPowerStatus(status) or status[0] != 1:
        raise RuntimeError('AC電源接続を確認できない')


def prepare_power(out: Path, info: dict) -> tuple[Path, subprocess.Popen]:
    state = out/'power_restore.json'
    original = info['original_plan']
    if power.active() != original:
        raise RuntimeError('確認後に電源プランが変わったため開始しない。CHECKからやり直す')
    value = dict(original=original, temporary=str(uuid.uuid4()), original_query=power.power('/qh', original),
                 controller_pid=os.getpid(), controller_birth=process_birth(os.getpid()), restored=False)
    power.write(state, value)
    power.write(ROOT/'pending.json', dict(state=str(state)))
    power.register_recovery(state)
    watcher = subprocess.Popen([sys.executable, '-B', str(HERE/'live_gate_power.py'),
        '--watch', str(os.getpid()), '--state', str(state)], creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(100):
        if state.with_suffix('.ready').exists():
            return state, watcher
        if watcher.poll() is not None:
            break
        time.sleep(.1)
    raise RuntimeError('独立復元監視の準備失敗。電源はまだ変更していない')


def make_low_plan(state: Path) -> dict:
    value = json.loads(state.read_text(encoding='utf-8'))
    temporary = value['temporary']
    power.power('/duplicatescheme', value['original'], temporary)
    for setting, index in ((power.BOOST, '0'), ('PROCTHROTTLEMIN', '5'), ('PROCTHROTTLEMAX', '100')):
        power.power('/setacvalueindex', temporary, power.SUB_PROCESSOR, setting, index)
    value['low_query'] = power.power('/qh', temporary)
    power.write(state, value)
    return value


def trial_args(out: Path, seconds: int) -> list[str]:
    return [sys.executable, '-B', '-u', str(HERE/'live_gate_trial.py'), '--bundle', str(BUNDLE),
        '--out', str(out), '--duration', str(seconds), '--start-sec', '0', '--startup-timeout', '1800',
        '--benchmark', '--quiet-throughout', '--port', '18855', '--video', str(VIDEO),
        '--ffmpeg-dll-dir', str(DLL), '--verified-runtime', str(RUNTIME), '--fast-audit',
        '--exact-seed', '--stdlib-cache', '--prepare-exact', '--deep-timing']


def trial(out: Path, name: str, info: dict) -> None:
    job_name = 'Local\\PuyoLiveGate_' + uuid.uuid4().hex
    mask = info['low_mask'] if name.endswith('low') else info['normal_mask']
    handle = create_job(job_name, mask)
    environment = dict(os.environ, LIVE_GATE_AUTHORIZED='ABBA-21C-FULL', LIVE_GATE_JOB=job_name,
        LIVE_GATE_MASK=str(mask), LIVE_GATE_CPU_RECEIPT=str(out/f'{name}.cpu.json'))
    try:
        with (out/f'{name}.log').open('wb') as log:
            child = subprocess.Popen(trial_args(out/name, info['video']['run_seconds']),
                stdout=log, stderr=subprocess.STDOUT, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
            code = child.wait()
        power.write(out/f'{name}.exit.json', dict(exit_code=code, at=time.time()))
        if code:
            raise RuntimeError(f'{name}が失敗。途中結果から係数を出さない')
    finally:
        kernel().CloseHandle(handle)


def run_trials(out: Path, info: dict, state: Path) -> None:
    settings = make_low_plan(state)
    for name in ORDER:
        check_idle(out)
        plan = settings['temporary'] if name.endswith('low') else settings['original']
        power.power('/setactive', plan)
        if power.active() != plan:
            raise RuntimeError('要求した電源プランを適用できない')
        power.write(out/f'{name}.power_applied.json', dict(plan=plan, at=time.time(), query=power.power('/qh', plan)))
        (out/'monitor.stop.run').write_text(name, encoding='utf-8')
        # 全条件で同じ待機時間。冷却と切替直後の過渡状態を揃える。
        time.sleep(COOLDOWN_SEC)
        trial(out, name, info)
        from live_gate_analysis import validate_flow
        power.write(out/name/'flow.json', validate_flow(out/name, info['video']['seconds']))


def execute(info: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT/time.strftime('run_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False)
    (out/'temp').mkdir()
    os.environ.update(TEMP=str(out/'temp'), TMP=str(out/'temp'), PYTHONDONTWRITEBYTECODE='1')
    api = kernel()
    if not api.SetProcessAffinityMask(api.GetCurrentProcess(), info['normal_mask']) or not api.SetPriorityClass(
            api.GetCurrentProcess(), NORMAL_PRIORITY):
        raise RuntimeError('controllerのCPU既定・Normalへの復帰失敗')
    check_idle(out)
    if digest(VIDEO) != VIDEO_SHA:
        raise RuntimeError('入力動画SHAが不一致')
    power.write(out/'protocol.json', info)
    state, watcher = prepare_power(out, info)
    monitor = None
    try:
        monitor = subprocess.Popen(['powershell', '-NoProfile', '-File', str(HERE/'monitor_live_gate.ps1'),
            '-Output', str(out/'environment.jsonl'), '-StopFile', str(out/'monitor.stop')],
            creationflags=subprocess.CREATE_NO_WINDOW)
        run_trials(out, info, state)
    finally:
        (out/'monitor.stop').touch()
        state.with_suffix('.stop').touch()
        watcher.wait(timeout=60)
        if monitor is not None:
            monitor.wait(timeout=60)
        restored = json.loads(state.read_text(encoding='utf-8'))
        if not restored.get('restored'):
            raise RuntimeError(f'復元を確認できない。cmd RESTOREを実行: {state}')
    from live_gate_analysis import analyze
    power.write(out/'coefficient.json', analyze(out, info))
    print(f'完了: {out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('CHECK', 'RESTORE', RUN_WORD))
    parser.add_argument('--confirm', help='非対話起動用の明示確認語。CHECKには作用しない')
    args = parser.parse_args()
    pending_recovery()
    if args.action is None:
        print('測定しません。CHECK=読取確認、RESTORE=前回復元、RUN-ABBA-21C=明示開始。')
        print('開始時に PC1-IDLE-FULL-ABBA を手入力。全長55分×4、約4時間。出力はD:。')
        return
    if args.action == 'RESTORE':
        print('未復元の電源プランはありません。')
        return
    info = preflight()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.action == 'CHECK':
        print('読取確認のみ。動画再生・測定・電源変更なし。')
        return
    confirmed = args.confirm
    if confirmed is None and sys.stdin.isatty():
        confirmed = input(f'PC1が空いていることを確認し {CONFIRM} を入力: ')
    if confirmed != CONFIRM:
        raise RuntimeError('確認語が一致しないため開始しない')
    execute(info)


if __name__ == '__main__':
    mutex = singleton()
    try:
        main()
    finally:
        kernel().CloseHandle(mutex)
