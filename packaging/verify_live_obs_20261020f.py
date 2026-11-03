"""再利用する実行系の全コードを検収し、同じ数値原票を再照合する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from obs_runtime_check import validate
import verify_live_obs_20261020e as parity

WORK = Path('D:/puyo_analyzer/packaging/build_20261020f_work')
BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020f_stage/PuyoLive')
RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))

if __name__ == '__main__':
    value = validate(BUNDLE, RUNTIME)
    (WORK/'runtime_verified.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print('既存実行系の全コード照合OK', flush=True)
    parity.previous.previous.guarded_run = parity.guarded_run
    parity.previous.main()
