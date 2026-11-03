"""既存の同値な実行系を読取り再利用するPC1本人用f版を組み立てる。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import build_live_obs_20261020e as previous

build = previous.build
build.BUILD = 'build_20261020f'
build.OUT = Path('D:/puyo_analyzer/packaging/build_20261020f_release')
build.BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020f_stage/PuyoLive')
build.FAST = build.BUNDLE
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261020e_release/'
    'PuyoLive_20261020e_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = Path('D:/PuyoLive_OBS_20261020f_USER_ONLY')
original_populate, original_guide = build.launchers.populate, build.launchers.guide
HELPERS = ('obs_runtime_check.py', 'obs_runtime_bootstrap.py', 'start_obs_limited.ps1')
HOOKS = {'python/sitecustomize.py', 'python/__pycache__/sitecustomize.cpython-312.pyc'}


def populate(folder: Path, bundle: Path, external: bool) -> None:
    """カメラ確認も含め、Pythonの初期import前にCPUと優先度を制限する。"""
    original_populate(folder, bundle, external)
    shutil.copy2(ROOT/'packaging/start_obs_limited.ps1', folder/'start_obs_limited.ps1')
    for name, extra in (('Start_PuyoLive_USER_ONLY.cmd', ''), ('List_Cameras.cmd', ' -Camera')):
        command = ('powershell -NoProfile -File "%~dp0start_obs_limited.ps1" '
                   f'-Bundle "{bundle}" -Entry "%~dp0."{extra}')
        build.launchers.cmd(folder/name, ['@echo off', 'setlocal',
            'echo USER ONLY - DO NOT DISTRIBUTE', command, 'if errorlevel 1 pause'])


def guide(*args: object, **kwargs: object) -> str:
    return original_guide(*args, **kwargs)+_public_dev_text('\nf版のPC1起動: 既存の<work>\\build_20261019a_stage\\PuyoLive\\pythonを読取りだけで再利用します。\nC:へ本体・キャッシュ・記録を新規作成せず、旧実行系のファイルも変更しません。\n起動ごとに全実行系を照合します。ソース・DLL・EXEは完全一致が必須です。\npycだけは保存ヘッダーとコード内ファイル名を除き、命令・定数・行情報を全て比較します。\n相違があれば起動を中止します。実際のアプリ・モデルは今回のD:配布本体だけを使います。\n既存実行系が無い場合は同梱Pythonを使います。SMB上では初期化が長くなる可能性があります。\nStartとList_Camerasの両入口は、Python起動前からCPU12〜15・BelowNormalを設定します。\n初期検収の原票はこの入口のoutput/startup_日時、実行の原票はoutput/実行日時です。\n本体smokeと速度門の結果はREPORTを参照してください。未合格を合格とは扱いません。\n')


def refresh() -> None:
    build.refresh()
    for name in HELPERS:
        shutil.copy2(ROOT/'packaging'/name, build.BUNDLE/name)
    inventory = json.loads((build.OLD_ZIP.parent/'files_sha256.json').read_text(encoding='utf-8'))
    files = {name.removeprefix('PuyoLive/python/'): digest for name, digest in inventory.items()
             if name.startswith('PuyoLive/python/') and name.removeprefix('PuyoLive/') not in HOOKS}
    value = dict(files=files, excluded_new_hook=sorted(HOOKS),
                 source_zip=str(build.OLD_ZIP), schema=1)
    (build.BUNDLE/'obs_runtime_manifest.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(f'既存実行系の照合目録: {len(files)} files', flush=True)


build.launchers.populate = populate
build.launchers.guide = guide

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('refresh', 'seal'))
    args = parser.parse_args()
    (refresh if args.action == 'refresh' else build.seal)()
