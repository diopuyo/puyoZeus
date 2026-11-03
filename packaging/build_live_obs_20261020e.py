"""D:の起動時ZIP読取りを減らす、出力同値の本人試験候補を組み立てる。"""
from pathlib import Path
from functools import partial
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import build_live_obs_20261020d as previous

build = previous.build
build.BUILD = 'build_20261020e'
build.OUT = Path('D:/puyo_analyzer/packaging/build_20261020e_release')
build.BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020e_stage/PuyoLive')
build.FAST = build.BUNDLE
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261020d_release/'
    'PuyoLive_20261020d_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = Path('D:/PuyoLive_OBS_20261020e_USER_ONLY')
CACHE_FILES = ('python/sitecustomize.py', 'python/__pycache__/sitecustomize.cpython-312.pyc')
original_populate = build.launchers.populate
original_guide = build.launchers.guide
original_packaged_files = build.packaged_files
PROGRESS_INTERVAL = 1000


def archive_progress(done: int, total: int) -> None:
    """SMBでの全件照合・格納の進捗だけを標準出力へ残す。"""
    if done % PROGRESS_INTERVAL == 0 or done == total:
        print(f'ZIP bytes verified: {done}/{total}', flush=True)


def populate(folder: Path, bundle: Path, external: bool) -> None:
    original_populate(folder, bundle, external)
    for name in ('Start_PuyoLive_USER_ONLY.cmd', 'List_Cameras.cmd'):
        path = folder/name
        lines = path.read_text(encoding='ascii').splitlines()
        lines[2:2] = ['set PUYO_OBS_STDLIB_CACHE=1', 'set PYTHONDONTWRITEBYTECODE=1']
        lines = [line.replace('"python\\python.exe" ', '"python\\python.exe" -B ') for line in lines]
        build.launchers.cmd(path, lines)


def guide(*args: object, **kwargs: object) -> str:
    return original_guide(*args, **kwargs)+'''
e版の起動補助: 凍結したPython標準ZIPを各processで一度だけ読みます。
元のzipimportへ同じbytesを渡し、コードの展開・検証・実行方法は変えません。
保持は1ファイル・最大16MiB、今回のPython 3.12標準ZIPは約3.65MiBです。
この入口だけPUYO_OBS_STDLIB_CACHE=1を設定します。製品本線の既定はOFFです。
Pythonは-Bで起動し、配布本体へ実行時のpycを書きません。記録・一時ファイルはD:です。
'''


def refresh() -> None:
    """アプリはd版を継承し、起動補助だけを追加する。"""
    build.refresh()
    target = build.BUNDLE/CACHE_FILES[0]
    shutil.copy2(ROOT/'packaging/obs_stdlib_cache.py', target)
    build.precompile(build.PRECOMPILE_RUNTIME, target)
    print('標準ZIPキャッシュを追加', flush=True)


def packaged_files() -> list[str]:
    return sorted(set(original_packaged_files()) | set(CACHE_FILES))


build.launchers.populate = populate
build.launchers.guide = guide
build.packaged_files = packaged_files
build.write_bundle_zip = partial(build.write_bundle_zip, progress=archive_progress)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('refresh', 'seal'))
    args = parser.parse_args()
    (refresh if args.action == 'refresh' else build.seal)()
