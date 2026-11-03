"""CPU制限を持たない入口と、入力世代を揃えた本人試験版を新規封印する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
from functools import partial
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import build_live_obs_20261020b as build

build.BUILD = 'build_20261021a'
build.OUT = Path('D:/puyo_analyzer/packaging/build_20261021a_release')
build.BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261021a_stage/PuyoLive')
build.FAST = build.BUNDLE
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261020f_release/'
    'PuyoLive_20261020f_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = Path('D:/PuyoLive_OBS_20261021a_USER_ONLY')
build.PRELOAD_ARCHIVES = True
build.PRECOMPILE_RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))
build.write_bundle_zip = partial(build.write_bundle_zip, in_memory=True)
build.inventory = partial(build.inventory, preload=True)
original_populate, original_guide = build.launchers.populate, build.launchers.guide
HELPERS = ('obs_runtime_check.py', 'obs_runtime_bootstrap.py', 'start_obs.ps1')


def wrapper() -> str:
    """起動元のaffinity・優先度を保ち、試験版の同値最適化だけを有効にする。"""
    source = build.launchers.WRAPPER.replace('import ctypes', 'import os')
    begin, end = source.index('    kernel ='), source.index('    template =')
    source = source[:begin]+'''    os.environ['PUYO_OBS_CPU_ISOLATION'] = '0'
    os.environ['PUYO_OBS_AMA_IDLE'] = '0'
    os.environ['PUYO_OBS_FAST_AUDIT'] = '1'
    os.environ['PUYO_OBS_EXACT_SEED'] = '1'
'''+source[end:]
    return source.replace('    run.mkdir(parents=True, exist_ok=False)',
        "    run.mkdir(parents=True, exist_ok=False)\n"
        "    temporary = run/'temp'\n    temporary.mkdir()\n"
        "    os.environ.update(TEMP=str(temporary), TMP=str(temporary))")


def populate(folder: Path, bundle: Path, external: bool) -> None:
    original_populate(folder, bundle, external)
    shutil.copy2(ROOT/'packaging/start_obs.ps1', folder/'start_obs.ps1')
    # 旧名を直接開いても制限が復活しないよう、互換入口も同じ実装にする。
    shutil.copy2(ROOT/'packaging/start_obs.ps1', folder/'start_obs_limited.ps1')
    for name, extra in (('Start_PuyoLive_USER_ONLY.cmd', ''), ('List_Cameras.cmd', ' -Camera')):
        command = ('powershell -NoProfile -File "%~dp0start_obs.ps1" '
                   f'-Bundle "{bundle}" -Entry "%~dp0."{extra}')
        build.launchers.cmd(folder/name, ['@echo off', 'setlocal',
            'echo USER ONLY - DO NOT DISTRIBUTE', command, 'if errorlevel 1 pause'])


def guide(*args: object, **kwargs: object) -> str:
    return original_guide(*args, **kwargs)+'''
CPUの割当てとプロセス優先度は起動元の既定を使います。特定CPUへの固定は行いません。
StartとList_Camerasは共通の入口を使います。puyo_live.jsonが配布設定です。
PC1では既存の検収済みPythonを読取り再利用し、app・モデル・記録・一時ファイルはD:を使います。
既存実行系がなければ同梱Pythonを使います。起動時に全ファイルの一致を検証します。
Pythonは-Bで起動し、配布本体に実行時pycを書きません。
v4.9fの発火履歴は認識入力の世代で区切り、B3の採用開始だけでは消去しません。
本線の採用既定・モデル・更新間隔は変更していません。
'''


def refresh() -> None:
    build.refresh()
    for name in HELPERS:
        shutil.copy2(ROOT/'packaging'/name, build.BUNDLE/name)


build.launchers.WRAPPER = wrapper()
build.launchers.populate = populate
build.launchers.guide = guide

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('refresh', 'seal'))
    args = parser.parse_args()
    (refresh if args.action == 'refresh' else build.seal)()
