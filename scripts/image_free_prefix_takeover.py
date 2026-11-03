"""既存PC2の3経路を止めず、独立監督で移管した旧前方補完を検証・継続する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

IDS = ('c68_old_prefix', 'video_d02_old_prefix')
POLL_SECONDS = 5


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare(existing: Path, bundle: Path, output: Path) -> dict:
    from supervisor import source_manifest, sha
    from scripts.image_free_stream_merge import validate
    plan = read(bundle/'route_plan.json')
    validate(plan)
    if {j['id'] for j in plan['jobs'] if j['host']=='pc2' and j['id'] in IDS} != set(IDS):
        raise ValueError('移管先の所有経路が違います')
    gate = read(bundle/'gate.json')
    if source_manifest(bundle/'source') != gate['source_manifest']:
        raise ValueError('固定ソースSHA不一致')
    for name, digest in gate['model_sha256'].items():
        if sha(existing/name) != digest:
            raise ValueError('モデルSHA不一致: '+name)
    output.mkdir(parents=True, exist_ok=True)
    config = read(existing/'config.json')
    paths = {j['video_id']: j['path'] for j in config['videos']}
    jobs = read(bundle/'transfer.json')['routes']
    config.update(videos=[dict(jobs[n]['job'], host='pc2', path=paths.get(jobs[n]['job']['video_id'], jobs[n]['job']['path'])) for n in IDS],
        output=str(output), production_output=str(output), old_cache=str(output/'old_cache'),
        transfer_bundle=str(bundle), max_workers=2, host='pc2')
    config['protected_outputs'] = [str(existing), *config.get('protected_outputs', [])]
    for name in ('ui.joblib', 'digit_student.joblib'):
        shutil.copyfile(existing/name, output/name)
    save(output/'config.json', config)
    proof = output/'proof'
    proof.mkdir(exist_ok=True)
    for name in ('ui.joblib', 'digit_student.joblib'):
        shutil.copyfile(existing/name, proof/name)
    save(proof/'config.json', dict(config, output=str(proof)))
    return config


def supervise(config_path: Path, bundle: Path, proof: bool) -> None:
    """既存ワーカーも予約するCPU割当と、両OS空き6GiB監督を共用する。"""
    from scripts.image_free_parallel import Monitor, stop_worker
    from scripts.image_free_resources import ResourceBudget
    from supervisor import tick, guarded_tick, install_progress_guard
    config = read(config_path)
    output = Path(config['output'])
    worker = 'image_free_transfer_worker.py' if proof else 'image_free_audit_worker.py'
    routes = {j['id']: dict(job=j, config=str(config_path), source=str(bundle/'source'),
        output=str(output), worker_entry=str(bundle/'controls'/worker)) for j in config['videos']}
    state = dict(routes=routes, children={}, codes={}, revision_id='prefix-takeover-v1',
        started_at=time.time(), source_manifest=read(bundle/'gate.json')['source_manifest'])
    monitor, budget = Monitor(), ResourceBudget()
    import supervisor as control
    original_observe = ResourceBudget.observe
    control.GUARD_COUNTS = {}
    install_progress_guard(output)
    limit = 1 if proof else 2
    try:
        while True:
            guarded_tick(output, lambda: tick(output, config, state, monitor, budget, limit))
            if state['codes'] and any(state['codes'].values()):
                raise RuntimeError('移管ワーカー異常終了: '+str(state['codes']))
            if len(state['codes']) == len(routes) and not state['children']:
                break
            time.sleep(POLL_SECONDS)
    finally:
        for child in state['children'].values():
            stop_worker(child)
        ResourceBudget.observe = original_observe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    sys.path.insert(0, str(bundle/'controls'))
    sys.path.insert(0, str(bundle/'source'))
    import fcntl
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with (output/'takeover.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (output/'config.json').exists():
            prepare(args.existing.resolve(), bundle, output)
        config = read(output/'config.json')
        receipts = [output/'proof'/n/'receipt.json' for n in IDS]
        if not all(p.exists() and read(p).get('passed') for p in receipts):
            supervise(output/'proof/config.json', bundle, True)
        if not all(p.exists() and read(p).get('passed') for p in receipts):
            raise ValueError('移送後の一致確認未完了')
        if args.verify_only:
            return
        supervise(output/'config.json', bundle, False)
        save(output/'takeover_complete.json', dict(passed=True, routes=IDS, at=time.time()))


if __name__ == '__main__':
    main()
