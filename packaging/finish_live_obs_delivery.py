"""同値検収の後に封印と実配布本体のsmokeを逐次実行する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import verify_live_obs_20261020b as verify

WORK = Path('D:/puyo_analyzer/packaging/build_20261020d_work')
BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020d_stage_r3/PuyoLive')
WAIT_SECONDS = 10
WAIT_LIMIT_SECONDS = 2*60*60


def await_parity(name: str) -> None:
    """未完成の終了票を成功として読まず、失敗なら配布処理へ進まない。"""
    path = WORK/f'{name}.exit.json'
    deadline = time.monotonic()+WAIT_LIMIT_SECONDS
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(WAIT_SECONDS)
            continue
        if value['exit_code']:
            raise RuntimeError(f'同値検収が未合格: {name}')
        return
    raise TimeoutError('同値検収の終了票待ちが2時間を超えました')


def run(out: Path, name: str, command: list[str]) -> None:
    print('START', name, flush=True)
    with (out/f'{name}.log').open('x', encoding='utf-8') as log:
        code = verify.guarded_run(command, dict(os.environ), log)
    if code:
        raise RuntimeError(f'{name}: exit={code}')
    print('DONE', name, flush=True)


def verify_sealed_assets(out: Path) -> None:
    """実行系・モデル・本線既定の意図しない変更を封印票から拒否する。"""
    release = Path('D:/puyo_analyzer/packaging/build_20261020d_release')
    receipt = json.loads((release/'archive_receipt.json').read_text(encoding='utf-8'))
    if not receipt.get('all_archive_members_byte_equal'):
        raise RuntimeError('ZIPと起動本体の全bytes照合が未完了です')
    unchanged = ('PuyoLive/python/', 'PuyoLive/app/models/', 'PuyoLive/LICENSES/')
    changes = receipt['changes']
    unexpected = [name for name in changes['changed']+changes['added']+changes['removed']
        if name.startswith(unchanged) or name.endswith(('.exe', '.dll'))
        or name == 'PuyoLive/app/src/production_config.py']
    if unexpected:
        raise RuntimeError(f'実行系・モデル・本線既定に予期しない差分: {unexpected}')
    (out/'sealed_assets.json').write_text(json.dumps(dict(
        passed=True, zip=receipt['zip'], sha256=receipt['sha256'],
        unchanged_prefixes=unchanged), indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='delivery_r1')
    parser.add_argument('--parity', default='parity_delivery')
    parser.add_argument('--cpu-isolation', action='store_true')
    args = parser.parse_args()
    import build_live_obs_20261020d as delivery
    expected = f"os.environ['PUYO_OBS_CPU_ISOLATION'] = '{int(args.cpu_isolation)}'"
    if expected not in delivery.build.launchers.WRAPPER:
        raise RuntimeError('配布入口とsmokeのCPU分離設定が一致しません')
    out = WORK/args.name
    out.mkdir(exist_ok=False)
    await_parity(args.parity)
    python = sys.executable
    builder = 'packaging/build_live_obs_20261020d.py'
    run(out, 'refresh', [python, '-u', builder, 'refresh'])
    run(out, 'seal', [python, '-u', builder, 'seal'])
    verify_sealed_assets(out)
    smoke = out/'zip_smoke'
    command = [python, '-u', 'packaging/smoke_live_obs_20261020.py',
        '--bundle', str(BUNDLE), '--out', str(smoke), '--duration', '240',
        '--startup-timeout', '1800', '--video', 'C:/puyo1_tools/soak/media/video_d04.mp4',
        '--start-sec', '600', '--ffmpeg-dll-dir',
        _public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'), '--port', '18839',
        '--fast-audit', '--exact-seed']
    if args.cpu_isolation:
        command.append('--cpu-isolation')
    run(out, 'zip_smoke', command)
    run(out, 'dom', ['node', 'packaging/check_live_obs_dom.cjs', str(smoke)])
    run(out, 'analysis', [python, 'packaging/analyze_live_obs_20261020d.py',
        str(smoke), str(smoke/'analysis.json')])
    (out/'receipt.json').write_text(json.dumps(dict(
        passed=True, bundle=str(BUNDLE), smoke=str(smoke),
        parity=args.parity, cpu_isolation=args.cpu_isolation), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
