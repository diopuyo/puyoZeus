"""移送したCPを既存全出力300fと照合し、一致時だけ現地指紋で継続可能にする。"""
from __future__ import annotations

import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

PROOF_FRAMES = 300
MID_FRAMES = 150


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def imported_state(snapshot: Path, output: Path, signature: str, key: str, config: dict) -> dict:
    from joblib.externals import cloudpickle
    from image_free_audit_checkpoint import rebind_cache
    metadata = read(snapshot/'checkpoint.json')
    with (snapshot/metadata['state']).open('rb') as file:
        state = cloudpickle.load(file)
    assert state['frame'] == metadata['frame']
    rebind_cache(state, Path(config['old_cache']), key)
    state['fingerprint'] = signature
    return state


def validate_assets(job: dict, config: dict, output: Path, snapshot: Path) -> None:
    original, local = read(snapshot/'old_cache_identity.json'), read(output/'old_cache_identity.json')
    for name in ('model_sha', 'video_sha', 'execution_control_sha256'):
        if original[name] != local[name]:
            raise ValueError('移送元と現地の資産SHA不一致: '+name)
    spec = read(Path(config['transfer_bundle'])/'transfer.json')['routes'][job['id']]
    digest = hashlib.sha256((snapshot/spec['state']).read_bytes()).hexdigest()
    if digest != spec['state_sha256']:
        raise ValueError('移送CPのSHA不一致')


def promote(job: dict, config: dict, snapshot: Path, output: Path, state: dict) -> None:
    from scripts.image_free_cache import write_chunk
    from scripts.compare_image_free_all import save_json
    target = Path(config['production_output'])/job['id']
    target.mkdir(parents=True, exist_ok=False)
    cp = read(snapshot/'checkpoint.json')
    for name in cp['chunks']:
        shutil.copyfile(snapshot/name, target/name)
    verified = read(output/'checkpoint.json')
    for name in verified['chunks']:
        shutil.copyfile(output/name, target/name)
    state['processed_frames'] += PROOF_FRAMES
    write_chunk(target/verified['state'], state)
    save_json(target/'checkpoint.json', dict(verified, chunks=cp['chunks']+verified['chunks']))
    save_json(target/'transfer_receipt.json', dict(frame=state['frame'], fingerprint=state['fingerprint'],
        source_snapshot=str(snapshot), preceding_chunks_preserved=True, verified_frames=PROOF_FRAMES))


def run(job: dict, config: dict, guard: Any) -> None:
    from scripts import image_free_stream as stream
    from scripts.image_free_timing import Timings
    from image_free_history_proof import calculate, differences
    output = Path(config['output'])/job['id']
    if (output/'checkpoint.json').exists():
        raise RuntimeError('中断済みの移管検証です。証拠を保全し、新しいOUTPUTで再検証してください')
    snapshot = Path(config['transfer_bundle'])/'snapshots'/job['id']
    signature, key = stream.fingerprint(job, config, output)
    validate_assets(job, config, output, snapshot)
    state = imported_state(snapshot, output, signature, key, config)
    start = state['frame']
    timing = Timings()
    first = calculate(job, state, start+MID_FRAMES, timing, guard)
    stream.comparison.checkpoint(output, state, first)
    state = stream.restore(job, config, output, signature, key)
    second = calculate(job, state, start+PROOF_FRAMES, timing, guard)
    stream.comparison.checkpoint(output, state, second)
    actual = first+second
    with gzip.open(snapshot/'reference.json.gz', 'rt') as file:
        expected = json.load(file)
    bad = differences(expected, actual)
    receipt = dict(passed=not bad and len(actual)==PROOF_FRAMES, frames=len(actual),
        start=start, end=state['frame'], mid_checkpoint=start+MID_FRAMES,
        mismatches=bad, float_comparison='IEEE754 binary64 bits', timings=timing.snapshot())
    stream.comparison.save_json(output/'receipt.json', receipt)
    if not receipt['passed']:
        raise ValueError('CP移送の全項目照合失敗。元CPは保持、現地本番は起動禁止')
    promote(job, config, snapshot, output, state)
    del state, first, second, actual
    gc.collect()
    stream.comparison.save_json(output/'progress.json', dict(frames=start+PROOF_FRAMES,
        complete=True, processed_frames=PROOF_FRAMES, pid=__import__('os').getpid()))


def main() -> None:
    source = Path(sys.argv[sys.argv.index('--source')+1])
    sys.path.insert(0, str(source))
    from scripts import image_free_stream as stream, compare_image_free_all as comparison
    from image_free_audit_worker import main as guarded_worker
    with comparison.mode(False), patch.object(stream, 'run', run):
        guarded_worker()


if __name__ == '__main__':
    main()
