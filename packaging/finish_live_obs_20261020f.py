"""f版の数値同値・封印・配布本体smokeを逐次実行し、失敗票も保存する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from finish_live_obs_delivery import run
import build_live_obs_20261020f as delivery

WORK = Path('D:/puyo_analyzer/packaging/build_20261020f_work')
RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))


def verify_assets(out: Path) -> None:
    receipt = json.loads((delivery.build.OUT/'archive_receipt.json').read_text(encoding='utf-8'))
    assert receipt['all_archive_members_byte_equal']
    prefixes = ('PuyoLive/python/', 'PuyoLive/app/', 'PuyoLive/LICENSES/')
    changes = receipt['changes']
    protected = [name for group in changes.values() for name in group if name.startswith(prefixes)]
    assert not protected, protected
    inventory = json.loads((delivery.build.OUT/'files_sha256.json').read_text(encoding='utf-8'))
    for name in delivery.HELPERS:
        expected = inventory['PuyoLive/'+name]
        assert sha256((ROOT/'packaging'/name).read_bytes()).hexdigest() == expected, name
        assert sha256((delivery.build.BUNDLE/name).read_bytes()).hexdigest() == expected, name
    (out/'sealed_assets.json').write_text(json.dumps(dict(passed=True,
        sha256=receipt['sha256'], unchanged_prefixes=prefixes), indent=2), encoding='utf-8')


def without_seconds(value: object) -> object:
    if isinstance(value, dict):
        return {k: without_seconds(v) for k, v in value.items() if k != 'seconds'}
    if isinstance(value, list):
        return [without_seconds(v) for v in value]
    return value


def verify_saved_parity() -> None:
    """封印だけの再開でも、完了した原票と既定OFFを全件確認する。"""
    reference = json.loads((ROOT/'docs/verification/live_obs_20261020d/'
        'delivery_parity_receipts.json').read_text(encoding='utf-8'))
    receipts = {name: json.loads((WORK/'parity'/name).read_text(encoding='utf-8'))
                for name in reference['receipts']}
    assert without_seconds(receipts) == without_seconds(reference['receipts'])
    digest = sha256((WORK/'parity/off.jsonl').read_bytes()).hexdigest()
    assert digest == reference['off_sha256']
    proof = dict(passed=True, work=str(WORK/'parity'), off_sha256=digest, receipts=receipts)
    (WORK/'parity_receipts.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')


def main(args: argparse.Namespace) -> None:
    out = WORK/args.name
    out.mkdir(exist_ok=False)
    python = sys.executable
    os.environ['PUYO_OBS_STDLIB_CACHE'] = '1'
    builder = 'packaging/build_live_obs_20261020f.py'
    if not args.resume_seal:
        run(out, 'refresh', [python, '-B', '-u', builder, 'refresh'])
        run(out, 'parity', [python, '-B', '-u', 'packaging/verify_live_obs_20261020f.py',
            '--out', str(WORK/'parity'), '--skip-tests'])
    verify_saved_parity()
    run(out, 'seal', [python, '-B', '-u', builder, 'seal'])
    verify_assets(out)
    smoke = out/'zip_smoke'
    run(out, 'zip_smoke', [python, '-B', '-u', 'packaging/smoke_live_obs_20261020.py',
        '--bundle', str(delivery.build.BUNDLE), '--out', str(smoke),
        '--verified-runtime', str(RUNTIME), '--duration', '240', '--startup-timeout', '1800',
        '--video', 'C:/puyo1_tools/soak/media/video_d04.mp4', '--start-sec', '600',
        '--ffmpeg-dll-dir', _public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'),
        '--port', '18841', '--fast-audit', '--exact-seed'])
    run(out, 'dom', ['node', 'packaging/check_live_obs_dom.cjs', str(smoke)])
    run(out, 'analysis', [python, '-B', 'packaging/analyze_live_obs_20261020d.py',
        str(smoke), str(smoke/'analysis.json')])
    (out/'receipt.json').write_text(json.dumps(dict(passed=True,
        bundle=str(delivery.build.BUNDLE), smoke=str(smoke), runtime=str(RUNTIME),
        parity=str(WORK/'parity'), cpu_isolation=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='delivery_r1')
    parser.add_argument('--resume-seal', action='store_true')
    options = parser.parse_args()
    try:
        main(options)
    except BaseException as error:
        (WORK/(options.name+'.failure.json')).write_text(json.dumps(dict(
            exception=type(error).__name__, message=str(error), stack=traceback.format_exc()),
            ensure_ascii=False, indent=2), encoding='utf-8')
        raise
