"""配布物 (PuyoLive/) の組み立て。Windows 上で実行する想定。出力先は D: (C: は容量逼迫)。

    python packaging/build_bundle.py --download   # 埋め込み Python と wheel を D: に取得
    python packaging/build_bundle.py              # 組み立て (bundle 直下に PuyoLive/ を作る)
    python packaging/build_bundle.py --zip        # 併せて zip 化

構成: PuyoLive/{PuyoLive.bat, puyo_live.example.json, README_ja.txt, LICENSES/,
                python/ (埋め込み Python 3.12 + CPU 版 wheel), app/ (ソース・モデル・設定・MANIFEST.json)}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlretrieve
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_closure import internal_closure, package_init_files  # noqa: E402


def _load_manifest_module() -> object:
    """launcher_manifest は標準ライブラリのみ。src.phase_j パッケージ (jsonschema 依存) を経由せず
    パスから直接読み込み、ビルド用 Python に依存を要求しない。"""
    import importlib.util
    path = ROOT / 'src' / 'phase_j' / 'launcher_manifest.py'
    spec = importlib.util.spec_from_file_location('launcher_manifest_standalone', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass が module を引くため登録が要る
    spec.loader.exec_module(module)
    return module


_manifest = _load_manifest_module()
build_manifest, installed_versions = _manifest.build_manifest, _manifest.installed_versions

PACKAGING_DIR = Path(__file__).resolve().parent
DEFAULT_WORK = Path('D:/puyo_analyzer/packaging')
DEFAULT_ASSET_ROOT = Path('C:/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer')
BUNDLE_NAME = 'PuyoLive'
PYTHON_VERSION = '3.12.3'  # 開発 venv (WSL) と同一
EMBED_URL = f'https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip'
TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
PYPI_INDEX = 'https://pypi.org/simple'
ENTRY_MODULES = ('scripts.run_live_pipeline_20260928', 'src.phase_j.launcher')
ASSET_LIST = PACKAGING_DIR / 'bundle_assets.txt'
SITE_PACKAGES = Path('Lib/site-packages')
PTH_LINES = ('python312.zip', '.', '..\\app', 'Lib\\site-packages', 'import site')
LICENSE_PATTERNS = ('LICEN*', 'COPYING*', 'NOTICE*')
SKIP_DIRS = frozenset({'__pycache__'})
PRUNE_DIR_NAMES = frozenset({'tests'})
PRUNE_PATHS = ('torch/include',)
# torch/lib の *.lib は C++ 拡張をリンクする時だけ必要 (dnnl.lib 653MB 等、実測で torch の約 7 割)
PRUNE_FILE_GLOBS = ('torch/lib/*.lib',)
FINISHED_MARK = '.puyo_finished'
DEFAULT_VC_RUNTIME_DIR = Path('C:/Windows/System32')  # VC++ 再頒布可能パッケージが入れた DLL の取得元
# Microsoft の再頒布可能コード (VC++ 2015-2022 CRT/C++ 標準ライブラリ)。packaging/scan_dll_deps.py の走査で
# 必須と出たのは msvcp140.dll (torch)。同系の補助 DLL も先回りで同梱する。
VC_RUNTIME_DLLS = ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll',
                   'msvcp140_2.dll', 'concrt140.dll')
ONNXRUNTIME_REQUIREMENT = 'onnxruntime==1.30.0'  # 2026-09-30 の合否評価 (docs/PHASE_J_ONNX_PREREGISTRATION) と同版
ONNXRUNTIME_DEPENDENCIES = ('flatbuffers', 'protobuf', 'packaging')
ONNX_WHEELHOUSE = Path('downloads/wheelhouse_onnx')
ONNX_WHEEL_PREFIXES = ('onnxruntime-', 'flatbuffers-', 'protobuf-', 'packaging-')
ONNX_SITE_GLOBS = ('onnxruntime', 'onnxruntime-*.dist-info', 'flatbuffers', 'flatbuffers-*.dist-info', 'google',
                   'protobuf-*.dist-info', 'packaging', 'packaging-*.dist-info')
SHIPPED_TEXT =('puyo_live.example.json', 'README_ja.txt', 'PuyoLive.bat')


def download_inputs(work: Path) -> None:
    """埋め込み Python と依存 wheel (torch は CPU 版) を取得する。既に有れば再取得しない。"""
    downloads = work / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    embed = downloads / Path(EMBED_URL).name
    if not embed.exists():
        urlretrieve(EMBED_URL, embed)
    wheelhouse = downloads / 'wheelhouse'
    if not any(wheelhouse.glob('*.whl')):
        subprocess.run([sys.executable, '-m', 'pip', 'download', '-r', str(PACKAGING_DIR / 'requirements-live.txt'),
                        '--only-binary=:all:', '--python-version', '3.12', '--platform', 'win_amd64',
                        '--index-url', TORCH_CPU_INDEX, '--extra-index-url', PYPI_INDEX,
                        '-d', str(wheelhouse)], check=True)
    fetch_onnx_inputs(work, downloads)


def fetch_onnx_inputs(work: Path, downloads: Path) -> None:
    """--with-onnx 用: ONNX Runtime (同梱) と、書き出しにだけ使う onnx (work/onnxtools。配布物には入れない)。
    puyo_core の Windows wheel は別途ビルドする (docs/PHASE_J_PACKAGING_PLAN §11-1、work/native_build/wheels)。"""
    onnx_house = downloads / 'wheelhouse_onnx'
    if not any(onnx_house.glob('onnxruntime-*.whl')):
        subprocess.run([sys.executable, '-m', 'pip', 'download', ONNXRUNTIME_REQUIREMENT, '--no-deps',
                        '--only-binary=:all:', '--python-version', '3.12', '--platform', 'win_amd64',
                        '-d', str(onnx_house)], check=True)
        for extra in ONNXRUNTIME_DEPENDENCIES:
            subprocess.run([sys.executable, '-m', 'pip', 'download', extra, '--no-deps', '--only-binary=:all:',
                            '--python-version', '3.12', '--platform', 'win_amd64', '-d', str(onnx_house)], check=True)
    tools = work / 'onnxtools'
    if not tools.is_dir():
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnx', 'ml_dtypes', '--no-deps', '--only-binary=:all:',
                        '--python-version', '3.12', '--platform', 'win_amd64', '--implementation', 'cp',
                        '--target', str(tools), '--no-compile'], check=True)


def stage_python(bundle: Path, work: Path) -> None:
    """埋め込み Python を展開し、._pth へアプリ・site-packages を登録して wheel を導入する。"""
    python = bundle / 'python'
    with zipfile.ZipFile(work / 'downloads' / Path(EMBED_URL).name) as archive:
        archive.extractall(python)
    (python / 'python312._pth').write_text('\n'.join(PTH_LINES) + '\n', encoding='ascii')
    # 推移依存 (torch→sympy 等) も含め、解決済みの wheelhouse を全件そのまま導入する。
    wheels = sorted(str(path) for path in (work / 'downloads' / 'wheelhouse').glob('*.whl'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--only-binary=:all:',
                    '--platform', 'win_amd64', '--python-version', '3.12', '--implementation', 'cp',
                    '--target', str(python / SITE_PACKAGES), '--no-compile', '--upgrade', *wheels], check=True)
    shutil.rmtree(python / SITE_PACKAGES / 'bin', ignore_errors=True)
    finish_python(bundle)


def prune_site_packages(python: Path) -> int:
    """実行に不要な物 (各ライブラリの tests/ と torch のヘッダ) を消し、削減バイト数を返す。"""
    site = python / SITE_PACKAGES
    targets = [path for path in site.rglob('*') if path.is_dir() and path.name in PRUNE_DIR_NAMES]
    targets += [site / rel for rel in PRUNE_PATHS if (site / rel).is_dir()]
    removed = 0
    for target in targets:
        if target.exists():
            removed += sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
            shutil.rmtree(target)
    for pattern in PRUNE_FILE_GLOBS:  # リンク用のインポートライブラリ (実行時は .dll だけ使う)
        for path in site.glob(pattern):
            removed += path.stat().st_size
            path.unlink()
    return removed


def precompile(python: Path, *targets: Path) -> None:
    """バイトコードを事前生成する。未生成だと初回起動が scipy.stats 等のコンパイルで 150 秒超になる (実測)。
    unchecked-hash は zip 展開で mtime がずれても無効化されない。配布物は凍結なので元ソースとの照合は不要。"""
    subprocess.run([str(python / 'python.exe'), '-m', 'compileall', '-q', '-f', '-j', '0',
                    '--invalidation-mode', 'unchecked-hash', *map(str, targets)], check=True)


def finish_python(bundle: Path) -> None:
    """python/ の仕上げ (不要物の削除→事前コンパイル)。済み印で二重実行を避ける。"""
    python = bundle / 'python'
    prune_site_packages(python)  # 安価で冪等なので毎回行う (削除対象の追加に追随できる)
    if (python / FINISHED_MARK).exists():
        return
    precompile(python, python / SITE_PACKAGES)  # 全 site-packages で数分かかるため済み印で 1 回に限る
    (python / FINISHED_MARK).write_text('precompiled\n', encoding='ascii')


def read_asset_list() -> list[str]:
    lines = ASSET_LIST.read_text(encoding='utf-8').splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]


def expand_asset(rel: str, asset_root: Path) -> list[str]:
    """資産 1 行 (ファイルまたはディレクトリ) を相対パス列へ。リポジトリ優先、無ければ asset_root。"""
    found: set[str] = set()
    for base in (ROOT, asset_root):
        target = base / rel
        if target.is_file():
            return [rel]
        if target.is_dir():  # ディレクトリは両方の和集合 (git 管理分と管理外分が分かれているため)
            found |= {path.relative_to(base).as_posix() for path in target.rglob('*')
                      if path.is_file() and not SKIP_DIRS & set(path.parts)}
    if not found:
        raise FileNotFoundError(f'資産が見つかりません: {rel} (リポジトリにも {asset_root} にも無い)')
    return sorted(found)


def source_of(rel: str, asset_root: Path) -> Path:
    for base in (ROOT, asset_root):
        if (base / rel).is_file():
            return base / rel
    raise FileNotFoundError(rel)


def collect_app_files(asset_root: Path) -> list[str]:
    """アプリ側に置く相対パス一覧: import 閉包の .py + 親 __init__.py + 資産リスト。"""
    rels: set[str] = set()
    for entry in ENTRY_MODULES:
        modules, _ = internal_closure(ROOT, entry)
        rels |= {path.relative_to(ROOT).as_posix() for path in modules.values()}
        rels |= {path.relative_to(ROOT).as_posix() for path in package_init_files(ROOT, modules)}
    for line in read_asset_list():
        rels |= set(expand_asset(line, asset_root))
    return sorted(rels)


def copy_app(bundle: Path, asset_root: Path) -> list[Path]:
    app = bundle / 'app'
    copied: list[Path] = []
    for rel in collect_app_files(asset_root):
        target = app / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_of(rel, asset_root), target)
        copied.append(target)
    (app / 'config' / 'device_calibration').mkdir(parents=True, exist_ok=True)  # 実行時に機器別較正を保存
    return copied


def copy_licenses(bundle: Path) -> int:
    """同梱ライブラリの LICENSE/NOTICE を LICENSES/<配布名>/ へ集める (件数を返す)。"""
    count = 0
    for info in (bundle / 'python' / SITE_PACKAGES).glob('*.dist-info'):
        for pattern in LICENSE_PATTERNS:
            for found in list(info.rglob(pattern)):
                if found.is_file():
                    target = bundle / 'LICENSES' / info.name[:-len('.dist-info')] / found.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(found, target)
                    count += 1
    return count


def write_manifest(bundle: Path, files: list[Path]) -> dict:
    app = bundle / 'app'
    packages = installed_versions(bundle / 'python' / SITE_PACKAGES)
    manifest = build_manifest(app, files, packages)
    (app / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return manifest


def make_zip(bundle: Path) -> Path:
    archive = shutil.make_archive(str(bundle), 'zip', root_dir=bundle.parent, base_dir=bundle.name)
    return Path(archive)


def prepare_bundle_dir(bundle: Path, work: Path, reuse_python: bool) -> None:
    """既定は全消しして Python から作り直す。reuse_python なら python/ だけ残す (pip 導入が約 12 分かかるため)。
    監査フック (sitecustomize.py) は配布物に入れないので、どちらの場合も残さない。"""
    if reuse_python and (bundle / 'python').is_dir():
        for child in bundle.iterdir():
            if child.name != 'python':
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        (bundle / 'python' / SITE_PACKAGES / 'sitecustomize.py').unlink(missing_ok=True)
        finish_python(bundle)
        return
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    stage_python(bundle, work)


def install_native_wheel(python: Path, wheel_dir: Path) -> str | None:
    """Windows 向けに別途ビルドした puyo_core (Rust) の wheel を展開する。無ければ None。
    本番構成は幽霊連鎖ルール ON で native を必須とする (Python フォールバックは未対応、実測で判明)。"""
    wheels = sorted(wheel_dir.glob('puyo_core-*.whl'))
    if not wheels:
        return None
    with zipfile.ZipFile(wheels[-1]) as archive:
        archive.extractall(python / SITE_PACKAGES)
    return wheels[-1].name


def add_vc_runtime(python: Path, source_dir: Path) -> dict[str, list[str]]:
    """再頒布可能な VC++ ランタイム DLL を python/ (アプリ ディレクトリ) へ置く。既存 (埋め込み Python が
    持つ vcruntime140*) は上書きしない。source_dir に無いものは missing に記録する。"""
    copied, kept, missing = [], [], []
    for name in VC_RUNTIME_DLLS:
        target = python / name
        if target.is_file():
            kept.append(name)
        elif (source_dir / name).is_file():
            shutil.copy2(source_dir / name, target)
            copied.append(name)
        else:
            missing.append(name)
    return dict(copied=copied, kept=kept, missing=missing)


def install_onnx_runtime(python: Path, work: Path) -> list[str]:
    """ONNX Runtime とその依存 wheel を展開する (ONNX バックエンド用。既定 OFF の機能の同梱)。"""
    names: list[str] = []
    for wheel in sorted((work / ONNX_WHEELHOUSE).glob('*.whl')):
        if wheel.name.startswith(ONNX_WHEEL_PREFIXES):
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(python / SITE_PACKAGES)
            names.append(wheel.name)
    return names


def remove_onnx_runtime(python: Path) -> None:
    """--with-onnx 無しのビルドに、前回の ONNX Runtime を持ち越さない (reuse-python 時)。"""
    site = python / SITE_PACKAGES
    for pattern in ONNX_SITE_GLOBS:
        for path in site.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def export_onnx_files(bundle: Path, work: Path) -> list[Path]:
    """app/models の CNN (.pt) を ONNX へ書き出し、生成物のパスを返す (MANIFEST 対象にするため)。
    書き出しには onnx パッケージが要る (配布物には入れず、作業用 work/onnxtools から一時的に読む)。"""
    code = ('import sys, runpy; sys.path[:0] = [%r, %r, %r]; sys.argv = [%r, %r]; runpy.run_path(%r, run_name="__main__")'
            % (str(work / 'onnxtools'), str(bundle / 'app'), str(PACKAGING_DIR.parent), 'export_onnx.py',
               str(bundle / 'app' / 'models'), str(PACKAGING_DIR / 'export_onnx.py')))
    subprocess.run([str(bundle / 'python' / 'python.exe'), '-c', code], check=True, cwd=bundle / 'app')
    return sorted((bundle / 'app' / 'models' / 'onnx').glob('*'))


def build(work: Path, asset_root: Path, make_archive: bool, reuse_python: bool = False,
          vc_runtime_dir: Path = DEFAULT_VC_RUNTIME_DIR, with_onnx: bool = False) -> dict:
    started = time.perf_counter()
    bundle = work / 'build' / BUNDLE_NAME
    prepare_bundle_dir(bundle, work, reuse_python)
    native = install_native_wheel(bundle / 'python', work / 'native_build' / 'wheels')
    vc_runtime = add_vc_runtime(bundle / 'python', vc_runtime_dir)
    onnx_wheels = install_onnx_runtime(bundle / 'python', work) if with_onnx else None
    if not with_onnx:
        remove_onnx_runtime(bundle / 'python')
    files = copy_app(bundle, asset_root)
    if with_onnx:
        files += export_onnx_files(bundle, work)
    precompile(bundle / 'python', bundle / 'app')
    files += sorted((bundle / 'app').rglob('*.pyc'))  # unchecked-hash のため MANIFEST で完全性を担保する
    for name in SHIPPED_TEXT:
        shutil.copy2(PACKAGING_DIR / name, bundle / name)
    licenses = copy_licenses(bundle)
    manifest = write_manifest(bundle, files)
    size = sum(path.stat().st_size for path in bundle.rglob('*') if path.is_file())
    result = dict(bundle=str(bundle), puyo_core_wheel=native, onnx_wheels=onnx_wheels, vc_runtime=vc_runtime, files_in_manifest=len(manifest['files']), license_files=licenses,
                  packages=len(manifest['packages']), bundle_bytes=size,
                  build_seconds=round(time.perf_counter() - started, 1))
    if make_archive:
        archive = make_zip(bundle)
        result.update(zip=str(archive), zip_bytes=archive.stat().st_size)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK, help='作業・出力ルート (既定 D:)')
    parser.add_argument('--asset-root', type=Path, default=DEFAULT_ASSET_ROOT,
                        help='git 管理外のモデル・データの置き場 (リポジトリに無い資産の取得元)')
    parser.add_argument('--download', action='store_true', help='埋め込み Python と wheel を取得して終了')
    parser.add_argument('--zip', action='store_true', help='組み立て後に zip 化')
    parser.add_argument('--reuse-python', action='store_true', help='python/ を作り直さず app 側だけ再構成')
    parser.add_argument('--with-onnx', action='store_true', help='ONNX Runtime と CNN の ONNX を同梱 (既定 OFF の切替用)')
    options = parser.parse_args()
    if options.download:
        download_inputs(options.work)
        return
    print(json.dumps(build(options.work, options.asset_root, options.zip, options.reuse_python,
                           with_onnx=options.with_onnx),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
