"""統合コードを読み取り参照する実行用ディレクトリをevalsetの新規ログに作る。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
INTEGRATED = Path('/mnt/d/puyo_analyzer/wt_integ_rt')
ASSETS = Path(_public_dev_text('<repo>/models/ui_templates'))
RUNTIME = ROOT/'logs/eval_set/falsedeath_fix/v2e/live_runtime'


def main() -> None:
    """認識テンプレを既存資産から参照し、較正の書込みは新規ログ内へ隔離する。"""
    RUNTIME.mkdir()
    for path in INTEGRATED.iterdir():
        if path.name in ('.git', 'models', 'config', 'logs'):
            continue
        (RUNTIME/path.name).symlink_to(path, target_is_directory=path.is_dir())
    (RUNTIME/'models').mkdir()
    for path in (INTEGRATED/'models').iterdir():
        if path.name != 'ui_templates':
            (RUNTIME/'models'/path.name).symlink_to(path, target_is_directory=path.is_dir())
    assert ASSETS.exists()
    (RUNTIME/'models/ui_templates').symlink_to(ASSETS, target_is_directory=True)
    shutil.copytree(INTEGRATED/'config', RUNTIME/'config')
    (RUNTIME/'logs').mkdir()
    (RUNTIME/'PROVENANCE.json').write_text(json.dumps(dict(integrated=str(INTEGRATED),
        templates=str(ASSETS), reason='統合側ui_templates欠落。既存資産を読み取り参照。閾値・構成変更なし。'),
        ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
