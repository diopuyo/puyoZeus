"""PC2自身の長時間CPで旧・新を照合し、途中復元も一致した場合だけ更新を許可する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

MIN_LATE_FRAME = 20000
PROOF_FRAMES = 180


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def snapshot(root: Path, output: Path, config: dict) -> list[dict]:
    """公開済みCP索引が指す不変ファイルをコピーし、実行中の本番には書き込まない。"""
    chosen = {}
    for job in config['videos']:
        path = root/job['id']/'checkpoint.json'
        if job['route'] in chosen or not path.exists():
            continue
        cp = read(path)
        if not MIN_LATE_FRAME <= cp['frame'] < job['end_frame']-PROOF_FRAMES:
            continue
        directory = output/job['id']
        directory.mkdir()
        shutil.copyfile(path.parent/cp['state'], directory/'before.pkl')
        save(directory/'checkpoint.json', dict(cp, state='before.pkl', chunks=[]))
        config['cases'][job['id']] = dict(seed=str(directory/'before.pkl'), seed_side=0, continuous_cp=True)
        chosen[job['route']] = job
    if set(chosen) != {'old', 'new'}:
        raise ValueError('2万フレーム経過後の旧・新CPが必要です')
    return list(chosen.values())


def prepare(root: Path, bundle: Path) -> tuple[dict, Path]:
    from image_free_audit_checkpoint import control_sha
    from image_free_audit_worker import FILES
    gate = read(bundle/'audit_gate.json')
    if control_sha([bundle/name for name in FILES]) != gate['control_sha256']:
        raise ValueError('更新一式の制御SHAが違います')
    output = root.parent/(root.name+'_history_preflight_'+str(time.time_ns()))
    output.mkdir()
    config = read(root/'config.json')
    prior = root/'controls/audit_gate.json'
    audit = prior.exists() and 'old' in read(prior).get('routes', [])
    config.update(output=str(output), old_cache=str(output/'old_cache'), cases={},
        baseline_audit=audit, control_sha256=gate['control_sha256'], predecessors=gate['predecessors'])
    config['protected_outputs'] = [str(root), *config.get('protected_outputs', [])]
    config['videos'] = snapshot(root, output, config)
    for name in ('ui.joblib', 'digit_student.joblib'):
        shutil.copyfile(root/name, output/name)
    save(output/'config.json', config)
    return config, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root, bundle = args.root.resolve(), Path(__file__).resolve().parent
    source = root/'source'
    sys.path.insert(0, str(source))
    from scripts import image_free_parallel as parallel
    from scripts.image_free_resources import ResourceBudget
    from supervisor import launch_route
    config, output = prepare(root, bundle)
    original = ResourceBudget.__init__
    measured = read(root/'supervisor.json')['measured_peak_rss']
    def initial(self: ResourceBudget, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        self.peak_rss, self.qualified = measured, True
    ResourceBudget.__init__ = initial
    def launch(options: argparse.Namespace, name: str, slot: int) -> object:
        route = dict(job={'id': name}, source=str(source), output=str(output),
            config=str(output/'config.json'), worker_entry=str(bundle/'image_free_history_worker.py'))
        child = launch_route(route)
        if child is None:
            raise RuntimeError('検証用の空きCPUがありません')
        return child
    parallel.launch = launch
    options = argparse.Namespace(config=output/'config.json', source=source, output=output, workers=1, auto_workers=True)
    if parallel.supervise(options) != 0:
        raise SystemExit('PC2独立照合失敗: 本番更新禁止')
    cases = {j['id']: read(output/j['id']/'receipt.json') for j in config['videos']}
    receipt = dict(passed=all(c['passed'] for c in cases.values()), cases=cases,
        control_sha256=config['control_sha256'], output=str(output), at=time.time())
    save(root/'audit_pc2_preflight.json', receipt)
    if not receipt['passed']:
        raise SystemExit('PC2で不一致: 本番更新禁止')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
