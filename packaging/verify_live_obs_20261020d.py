"""d版の保存原票同値・既定OFF・監査配送・CPU割当てを逐次検収する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import verify_live_obs_20261020b as previous

previous.OUT = Path('D:/puyo_analyzer/packaging/build_20261020d_work/parity')


def tests() -> None:
    # 配布のPython 3.12にはpytestを同梱しない。検収用の純Pythonパッケージだけを追加。
    sys.path.append(_public_dev_text('<home>/AppData/Local/Python/pythoncore-3.14-64/Lib/site-packages'))
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    import pytest
    code = pytest.main(['-q', 'tests/phase_j/test_live_obs_cpu.py',
        'tests/phase_j/test_live_ama_seed.py',
        'tests/phase_j/test_live_obs_audit.py', 'tests/phase_j/test_live_obs_20261020.py',
        'tests/phase_j/test_live_stage_timing.py', 'tests/phase_j/test_live_reference_times.py',
        'tests/phase_j/test_live_publish.py', 'tests/phase_j/test_worker_supervisor.py',
        'tests/test_live_b3_measurements.py', 'tests/test_live_score_cadence.py',
        'tests/test_live_b19.py::test_spool_failure_rolls_back_archive_and_replays',
        '--basetemp='+str(previous.OUT/'pytest_tmp'),
        '-o', 'cache_dir='+str(previous.OUT/'pytest_cache')])
    if code:
        raise RuntimeError(f'pytest: {code}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=previous.OUT)
    parser.add_argument('--skip-tests', action='store_true',
                        help='別ログで回帰検査を実施済みの場合、保存原票の同値確認だけを行う')
    parser.add_argument('--stages', nargs='+', choices=('off', 'ama_inputs', 'm3_live',
                        'v49f_80_7', 'v49f_51_4', 'ama_values', 'static'))
    args = parser.parse_args()
    previous.OUT = args.out
    previous.SELECTED = set(args.stages) if args.stages else None
    previous.OUT.mkdir(exist_ok=False)
    os.environ['PUYO_OBS_FAST_AUDIT'] = '1'
    os.environ['PUYO_OBS_EXACT_SEED'] = '1'
    if not args.skip_tests:
        tests()
    os.environ['PUYO_OBS_CPU_ISOLATION'] = '1'
    os.environ['PUYO_OBS_FAST_AUDIT'] = '1'
    if not previous.M3_RECORD.exists():
        previous.M3_RECORD = Path('D:/puyo_analyzer/b3_eval_records/attempts/'
                                 'b3_zenchi_s1/1791168806059117750/record')
        # 前回、研究版全列との一致を確認済みの原票を継承する。
        previous.M3_REFERENCE = Path('D:/puyo_analyzer/packaging/'
                                    'build_20261020d_work/parity_seed/m3_live.jsonl')
    previous.main()
    previous.run('static', 'verify_live_obs_static.py', [previous.OUT/'static.json'])


if __name__ == '__main__':
    main()
