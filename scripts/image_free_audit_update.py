"""合格票を照合し、自分の監督だけを停止して同じCPから制御を更新する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from image_free_audit_checkpoint import control_sha
from image_free_audit_worker import FILES

AUTHORIZATION = 'user-authorized-start-before-r2-2026-10-08'
STOP_TIMEOUT = 60


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def alive(pid: int) -> bool:
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1][0] != 'Z'


def stop_owned(root: Path, state: dict) -> None:
    pid = state.get('supervisor_pid')
    if not pid or not alive(pid):
        return
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    expected = str(root/'controls/supervisor.py').encode()
    if expected not in command or str(root/'config.json').encode() not in command:
        raise ValueError('監督PIDの実コマンドが更新対象と一致しません')
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+STOP_TIMEOUT
    while any(alive(p) for p in [pid, *state.get('worker_pids', {}).values()]):
        if time.monotonic() > deadline:
            raise TimeoutError('旧監督/子プロセスの終了を確認できません')
        time.sleep(1)


def install(root: Path, bundle: Path, start: bool) -> dict:
    gate = read(bundle/'audit_gate.json')
    if not gate['passed'] or gate['control_sha256'] != control_sha([bundle/n for n in FILES]):
        raise ValueError('制御SHA/合格票が不正です')
    if hashlib.sha256((bundle/'proof.json').read_bytes()).hexdigest() != gate['proof_sha256']:
        raise ValueError('段3証拠SHAが違います')
    if not read(bundle/'proof.json')['passed']:
        raise ValueError('段3未合格です')
    state = read(root/'supervisor.json') if (root/'supervisor.json').exists() else {}
    config = read(root/'config.json')
    if read(root/'gate.json')['source_manifest'] != gate['source_manifest']:
        raise ValueError('旧本体ソースが検証版と異なります')
    if start and config['host'] == 'pc2':
        local = read(root/'audit_pc2_preflight.json')
        if not local['passed'] or local.get('control_sha256') != gate['control_sha256']:
            raise ValueError('PC2で今回の制御SHAの独立再検証が必要です')
    before = {j['id']: read(root/j['id']/'checkpoint.json') for j in config['videos']
              if (root/j['id']/'checkpoint.json').exists()}
    stop_owned(root, state)
    controls = root/'controls'
    backup = root/'audit_upgrade_backup'
    backup.mkdir(exist_ok=True)
    for path in (root/'config.json', controls/'supervisor.py'):
        target = backup/path.name
        if not target.exists():
            shutil.copyfile(path, target)
    for name in (*FILES, 'audit_gate.json', 'proof.json', 'supervisor.py'):
        shutil.copyfile(bundle/name, controls/name)
    config['audit_worker'] = str(controls/'image_free_audit_worker.py')
    save(root/'config.json', config)
    receipt = dict(at=time.time(), before=before, control_sha256=gate['control_sha256'],
                   old_source_unchanged=True, resume_from_checkpoint=True, started=False)
    if start:
        command = [sys.executable, str(controls/'supervisor.py'), '--config', str(root/'config.json'),
                   '--source', str(root/'source'), '--gate', str(root/'gate.json'), '--execute', AUTHORIZATION]
        env = dict(os.environ, PYTHONPATH=str(root/'source'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root/'launch.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root, env=env, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(started=True, supervisor_pid=child.pid)
    save(root/'audit_upgrade.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    print(json.dumps(install(args.root.resolve(), Path(__file__).resolve().parent, args.start)))


if __name__ == '__main__':
    main()
