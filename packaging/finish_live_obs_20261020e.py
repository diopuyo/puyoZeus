"""e候補の同値・封印・配布本体smokeを逐次検収する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from finish_live_obs_delivery import run
import build_live_obs_20261020e as delivery

WORK = Path('D:/puyo_analyzer/packaging/build_20261020e_work')


def verify_assets(out: Path) -> None:
    """旧実行系は不変、追加する起動補助2ファイルだけを明示許可する。"""
    receipt = json.loads((delivery.build.OUT/'archive_receipt.json').read_text(encoding='utf-8'))
    assert receipt['all_archive_members_byte_equal']
    allowed = {'PuyoLive/'+name for name in delivery.CACHE_FILES}
    changes = receipt['changes']
    prefixes = ('PuyoLive/python/', 'PuyoLive/app/models/', 'PuyoLive/LICENSES/')
    protected = [name for name in changes['changed']+changes['removed']
                 if name.startswith(prefixes) or name.endswith(('.exe', '.dll'))
                 or name == 'PuyoLive/app/src/production_config.py']
    additions = {name for name in changes['added'] if name.startswith(prefixes)}
    assert not protected, protected
    assert additions == allowed, additions
    (out/'sealed_assets.json').write_text(json.dumps(dict(
        passed=True, sha256=receipt['sha256'], allowed_additions=sorted(allowed),
        unchanged_prefixes=prefixes), indent=2), encoding='utf-8')


def main() -> None:
    out = WORK/'delivery_r1'
    out.mkdir(exist_ok=False)
    python = sys.executable
    os.environ['PUYO_OBS_STDLIB_CACHE'] = '1'
    builder = 'packaging/build_live_obs_20261020e.py'
    run(out, 'refresh', [python, '-B', '-u', builder, 'refresh'])
    run(out, 'parity', [python, '-B', '-u', 'packaging/verify_live_obs_20261020e.py',
        '--out', str(WORK/'parity'), '--skip-tests'])
    run(out, 'seal', [python, '-B', '-u', builder, 'seal'])
    verify_assets(out)
    smoke = out/'zip_smoke'
    run(out, 'zip_smoke', [python, '-B', '-u', 'packaging/smoke_live_obs_20261020.py',
        '--bundle', str(delivery.build.BUNDLE), '--out', str(smoke),
        '--duration', '240', '--startup-timeout', '1800',
        '--video', 'C:/puyo1_tools/soak/media/video_d04.mp4', '--start-sec', '600',
        '--ffmpeg-dll-dir', _public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'),
        '--port', '18840', '--fast-audit', '--exact-seed', '--stdlib-cache'])
    run(out, 'dom', ['node', 'packaging/check_live_obs_dom.cjs', str(smoke)])
    run(out, 'analysis', [python, '-B', 'packaging/analyze_live_obs_20261020d.py',
        str(smoke), str(smoke/'analysis.json')])
    (out/'receipt.json').write_text(json.dumps(dict(
        passed=True, bundle=str(delivery.build.BUNDLE), smoke=str(smoke),
        parity=str(WORK/'parity'), cpu_isolation=False, stdlib_cache=True),
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
