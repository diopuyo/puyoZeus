"""PC2の更新も既知の同居版1080経路フレームと全項目照合してから許可する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), Path(__file__).resolve().parent
    source = root/'source'
    sys.path.insert(0, str(source))
    from scripts import image_free_parallel as parallel
    from scripts.image_free_resources import unoccupied_cpus
    from scripts.image_free_stream_proof import verify
    config = json.loads((root/'preflight_config.json').read_text())
    reference = Path(config['output'])
    output = root/'audit_preflight'
    output.mkdir(exist_ok=True)
    for name in ('ui.joblib', 'digit_student.joblib'):
        shutil.copyfile(reference/name, output/name)
    shutil.copytree(reference/config['baseline_id'], output/config['baseline_id'], dirs_exist_ok=True)
    config.update(output=str(output), old_cache=str(output/'old_cache'))
    path = root/'audit_preflight_config.json'
    parallel.save_json(path, config)
    def launch(options: argparse.Namespace, name: str, slot: int) -> subprocess.Popen:
        command = [sys.executable, str(bundle/'image_free_audit_worker.py'), '--config', str(path),
                   '--source', str(source), '--worker', name, '--parent', str(os.getpid()),
                   '--cpu', str(unoccupied_cpus(parallel.worker_cpus())[0])]
        directory = output/name
        directory.mkdir(exist_ok=True)
        with (directory/'parallel.log').open('a') as log:
            return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ, **parallel.THREAD_ENV))
    parallel.launch = launch
    options = argparse.Namespace(config=path, source=source, output=output, workers=2, auto_workers=True)
    if parallel.supervise(options) != 0:
        raise SystemExit('PC2検証の計算に失敗しました。本番は禁止です。')
    receipt = verify(config)
    parallel.save_json(root/'audit_pc2_preflight.json', receipt)
    if not receipt['passed']:
        raise SystemExit('PC2で全項目一致を再現できません。本番は禁止です。')
    state = parallel.read_json(output/'supervisor.json')
    parallel.save_json(root/'rss_seed.json', dict(source_manifest=parallel.read_json(root/'source_manifest.json'),
        measured_peak_rss=state['measured_peak_rss']))
    print('PC2監査更新: 1080経路フレーム全項目一致')


if __name__ == '__main__':
    main()
