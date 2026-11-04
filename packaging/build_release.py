"""本番配布先を使わず、署名なしインストーラー・zip・配布台帳を一括生成する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'packaging')]
from build_bundle import build, DEFAULT_ASSET_ROOT, write_manifest, precompile, LICENSE_PATTERNS
from bundle_archive import is_runtime_file, write_bundle_zip
from release_privacy import sanitize_bundle_provenance, refresh_model_hashes
from build_launchers import build_launchers

DEFAULT_OUTPUT = Path('D:/puyo_analyzer/b3_data/codex_tasks/dist_app_release')
INPUTS = Path('D:/puyo_analyzer/packaging')
MODEL = Path('D:/puyo_analyzer/share/results/pc2/fast_pipeline/runs/v6/final/model.joblib')
TABLE = Path('D:/puyo_analyzer/b3_data/timing_v1/timing_table.json')
WHEEL = Path('D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_v5swap/wheels/puyo_core-0.1.0-cp312-cp312-win_amd64.whl')
WHEEL_SHA = '7f631aadd7478b16a86d6c84fb7b1a3e44971c8b77be95bcb720959a66693bee'
VERSION = '0.6.0-preview.11'
MARKER = '.puyolive-release-workspace'
RC_MODELS = Path(_public_dev_text('<work>/build_20261101_obs_w2_local/stage/PuyoLive/app/models'))
# 7d19831b・Rust 1.97.1・24w2同一機能。個人パスをremapし240盤面と指紋9項目が一致。
RC_NATIVE = Path(_public_dev_text('<work>/codex_jobs/release_rc_build_20261102/native_remap/puyo_core.pyd'))
RC_NATIVE_SHA = 'd631e3d40cb493eeaf1ad58ce97f5e2d0db2a8823407d5f00617cd15744c136f'
ZIP_EPOCH_SECONDS = 315532800  # ZIPが表現できる1980-01-01以降へ時刻だけ補正する。
RC_MODEL_DIRS = ('b3_c2', 'b3_c2_ens', 'b3_ex_L200H_retR', 'ama_plain', 'live_ex', 'v49f')
WINDOW_WHEELS = ('pywebview==6.1', 'bottle==0.13.4', 'pythonnet==3.2.1',
                 'clr_loader==0.3.1', 'cffi==2.1.1', 'pycparser==3.12',
                 'typing_extensions==4.16.0')
PYTHON_RUNTIME_CONFIG = ('<configuration><runtime><loadFromRemoteSources enabled="true"/>'
                         '</runtime></configuration>\n')


def write_python_runtime_configs(bundle: Path) -> None:
    """エクスプローラーが引き継いだダウンロード印でも同梱DLLを読めるようにする。"""
    for executable in ('python.exe', 'pythonw.exe'):
        (bundle / 'python' / (executable + '.config')).write_text(PYTHON_RUNTIME_CONFIG, encoding='ascii')


def prepare_window_wheels(output: Path) -> None:
    """既存のcamera_wheelsがあっても窓の依存をすべて揃える。"""
    extra = output / 'camera_wheels'
    extra.mkdir(parents=True, exist_ok=True)
    if all(list(extra.glob(spec.replace('==', '-') + '-*.whl'))
           for spec in (*WINDOW_WHEELS, 'proxy_tools==0.1.0')):
        return
    subprocess.run([sys.executable, '-m', 'pip', 'download', *WINDOW_WHEELS,
                    '--no-deps', '--only-binary=:all:', '--platform', 'win_amd64',
                    '--python-version', '312', '--implementation', 'cp',
                    '--abi', 'cp312', '-d', str(extra)], check=True)
    # proxy_toolsはsdistのみ。純Pythonのwheelを配布作業フォルダ内で作る。
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', 'proxy_tools==0.1.0',
                    '--no-deps', '--wheel-dir', str(extra)], check=True)


def copy_window_licenses(bundle: Path) -> None:
    """wheelにない原文と、libffi・WebView2 SDKの通知を補う。"""
    source = ROOT / 'packaging/licenses/app_window'
    for directory in source.iterdir():
        if directory.is_dir():
            shutil.copytree(directory, bundle / 'LICENSES' / directory.name, dirs_exist_ok=True)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_rc_assets(bundle: Path, models: Path) -> dict[str, str]:
    """24w2の資産を同名で収録し、参照exeは入れない。"""
    hashes: dict[str, str] = {}
    for name in RC_MODEL_DIRS:
        for source in sorted((models / name).rglob('*')):
            if not source.is_file() or source.suffix in ('.exe', '.pyc'):
                continue
            relative = source.relative_to(models)
            target = bundle / 'app/models' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = digest(target)
    for relative in ('b3_c2_ens/ensemble_ref3_logit.json', 'b3_ex_L200H_retR/ensemble.json'):
        manifest = bundle / 'app/models' / relative
        members = json.loads(manifest.read_text(encoding='utf-8'))['members']
        if len(members) != 3:
            raise ValueError(f'3 seedではありません: {relative}')
        for member in members:
            if digest(manifest.parent / member['name']) != member['sha256']:
                raise ValueError(f'モデルSHA不一致: {relative}/{member["name"]}')
    shutil.copy2(models / 'b3/timing_table.json', bundle / 'app/models/b3/timing_table.json')
    hashes['b3/timing_table.json'] = digest(bundle / 'app/models/b3/timing_table.json')
    native = RC_NATIVE
    if digest(native) != RC_NATIVE_SHA:
        raise ValueError('パス除去済み24w2 nativeのSHA不一致')
    shutil.copy2(native, bundle / 'python/Lib/site-packages/puyo_core/puyo_core.cp312-win_amd64.pyd')
    return hashes


# 0.6.0 の補助別名との互換性を保つ。全原文は相対パス付きで別途保存する。
FLAT_LICENSE_SOURCES = {
    'numpy-2.4.4': {
        'license.md': 'licenses/numpy/fft/pocketfft/LICENSE.md',
        'license': 'licenses/numpy/_core/src/umath/svml/LICENSE',
    },
}


def copy_flat_release_licenses(bundle: Path) -> None:
    """平坦な補助別名の採用元を固定。原文は後で相対パス付きで全て保存する。"""
    site = bundle / 'python/Lib/site-packages'
    for info in sorted(site.glob('*.dist-info')):
        sources = {path for pattern in LICENSE_PATTERNS for path in info.rglob(pattern)
                   if path.is_file()}
        selected: set[str] = set()
        for source in sorted(sources, key=lambda path: path.relative_to(info).as_posix()):
            key = source.name.casefold()
            if key in selected:
                continue
            selected.add(key)
            preferred = FLAT_LICENSE_SOURCES.get(info.name.removesuffix('.dist-info'), {}).get(key)
            if preferred and (info / preferred).is_file():
                source = info / preferred
            target = bundle / 'LICENSES' / info.name.removesuffix('.dist-info') / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_release_licenses(bundle: Path) -> None:
    """同名の原文を元の相対ディレクトリ付きで保存する。"""
    copy_flat_release_licenses(bundle)
    site = bundle / 'python/Lib/site-packages'
    for info in sorted(site.glob('*.dist-info')):
        for source in sorted(info.rglob('*')):
            if source.is_file() and ('licenses' in source.relative_to(info).parts
                                    or source.name.upper().startswith(('LICEN', 'COPYING', 'NOTICE'))):
                relative = source.relative_to(info)
                if relative.parts[0] == 'licenses':
                    relative = Path(*relative.parts[1:])
                target = bundle / 'LICENSES' / info.name.removesuffix('.dist-info') / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    for name in ('LICENSE', 'ThirdPartyNotices.txt'):
        target = bundle / 'LICENSES/onnxruntime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(site / 'onnxruntime' / name, target)
    shutil.copytree(ROOT / 'native/puyo_core/licenses', bundle / 'LICENSES/puyo_core', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'packaging/licenses/release', bundle / 'LICENSES', dirs_exist_ok=True)
    copy_window_licenses(bundle)
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(ROOT / name, bundle / name)
    for target in (bundle / 'LICENSES').rglob('*'):
        if target.is_file() and target.stat().st_mtime < ZIP_EPOCH_SECONDS:
            os.utime(target, (ZIP_EPOCH_SECONDS, ZIP_EPOCH_SECONDS))


def prune_release_development_files(bundle: Path) -> None:
    """配布先でコンパイルしないための開発部品と追加wheelの試験を除く。"""
    site = bundle / 'python/Lib/site-packages'
    for directory in list(site.rglob('tests')):
        if directory.is_dir():
            shutil.rmtree(directory)
    development_suffixes = {'.lib', '.pdb', '.h', '.hpp', '.pyx', '.pxd',
                            '.c', '.cpp', '.f', '.f90', '.a', '.o', '.obj', '.cu'}
    for path in site.rglob('*'):
        if path.is_file() and path.suffix.lower() in development_suffixes:
            path.unlink()
    # 常駐版は描画せず、オフラインの認識描画だけで使う台本は含めない。
    scripts = bundle / 'app/scripts'
    (scripts / 'visualize_recognition.py').unlink(missing_ok=True)
    for path in (scripts / '__pycache__').glob('visualize_recognition.*.pyc'):
        path.unlink()
    # 既存のステージを再利用する場合も、旧い調整値のバックアップを残さない。
    backup = bundle / 'app/data/per_video_hsv_ranges/_backup_pre_expand_yellow'
    if backup.is_dir():
        if not backup.resolve().is_relative_to(bundle.resolve()):
            raise ValueError('バックアップが配布先の外を指しています')
        shutil.rmtree(backup)


def safe_output(output: Path) -> Path:
    output = output.resolve()
    forbidden = [INPUTS.resolve(), ROOT.resolve(), DEFAULT_ASSET_ROOT.resolve()]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in forbidden):
        raise ValueError('本番・入力・作業ツリーへは出力できません')
    if output.exists() and any(output.iterdir()) and not (output / MARKER).exists():
        raise ValueError('既存の空でない出力先は使えません')
    output.mkdir(parents=True, exist_ok=True)
    (output / MARKER).write_text('PuyoLive isolated release workspace\n', encoding='ascii')
    return output


def prepare_inputs(output: Path) -> None:
    """読取専用のキャッシュから複製。本番buildディレクトリは参照しない。"""
    for name in ('downloads',):
        target = output / name
        if not target.exists():
            shutil.copytree(INPUTS / name, target)
    wheels = output / 'native_build/wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    if digest(WHEEL) != WHEEL_SHA:
        raise ValueError('検証済みnative wheelと一致しません')
    shutil.copy2(WHEEL, wheels / WHEEL.name)
    extra = output / 'camera_wheels'
    if not list(extra.glob('pystray-*.whl')):
        subprocess.run([sys.executable, '-m', 'pip', 'download', 'cv2-enumerate-cameras==1.4.0', 'websocket-client==1.8.0', 'pystray==0.19.5', 'six==1.17.0',
            '--no-deps', '--only-binary=:all:', '-d', str(extra)], check=True)
    prepare_window_wheels(output)


def compiler() -> Path:
    paths = [Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Inno Setup 6/ISCC.exe',
             Path('C:/Program Files (x86)/Inno Setup 6/ISCC.exe')]
    found = next((p for p in paths if p.is_file()), None)
    if found is None:
        raise FileNotFoundError('Inno Setupが必要: winget install --id JRSoftware.InnoSetup --exact --scope user')
    return found


def retain_python_vc_runtime(bundle: Path, output: Path) -> None:
    """再利用ステージでもSystem32由来のDLLを除き、Python原本だけ残す。"""
    from build_bundle import VC_RUNTIME_DLLS
    with zipfile.ZipFile(output / 'downloads/python-3.12.3-embed-amd64.zip') as archive:
        originals = {name.lower(): name for name in archive.namelist()}
        for name in VC_RUNTIME_DLLS:
            target = bundle / 'python' / name
            if name in originals:
                target.write_bytes(archive.read(originals[name]))
            else:
                target.unlink(missing_ok=True)


def write_sendlogs(bundle: Path) -> None:
    """旗に応じた宛先を単一の設定から取り、zipとインストーラーで共有する。"""
    from src.phase_j.distribution_config import feedback_site_url
    from src.phase_j.distribution_logs import RELAY_PLACEHOLDER
    script = (ROOT / 'packaging/SendLogs.ps1').read_text(encoding='ascii')
    site = feedback_site_url()
    if site:
        endpoint = (site + "/api/reports").replace("'", "''")
        script = script.replace(RELAY_PLACEHOLDER, endpoint)
    (bundle / 'tools').mkdir(exist_ok=True)
    (bundle / 'tools/SendLogs.ps1').write_text(script, encoding='ascii')


def arrange_launchers(bundle: Path) -> None:
    """直下の起動・削除入口を各1つにし、互換入口はtoolsへ置く。"""
    tools = bundle / 'tools'
    tools.mkdir(exist_ok=True)
    keep = set()
    for source in bundle.iterdir():
        if source.suffix.lower() not in ('.bat', '.vbs') or source.name in keep:
            continue
        target = tools / source.name
        if target.exists():
            raise FileExistsError(target)
        content = source.read_text(encoding='utf-8-sig')
        if source.suffix.lower() == '.bat':
            content = content.replace('%~dp0', '%~dp0..\\')
        else:
            content = content.replace('fs.GetParentFolderName(WScript.ScriptFullName)',
                                      'fs.GetParentFolderName(fs.GetParentFolderName(WScript.ScriptFullName))')
        target.write_text(content, encoding='utf-8')
        source.unlink()


def arrange_release_documents(bundle: Path) -> None:
    """説明書は日英併記の1つとし、補助ファイルをtoolsへまとめる。"""
    tools = bundle / 'tools'
    tools.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'packaging/puyo_live.example.json', tools / 'puyo_live.example.json')
    shutil.copy2(ROOT / 'docs/mascot_expressions_preview.png', tools / 'mascot_expressions_preview.png')
    shutil.copy2(ROOT / 'docs/DISTRIBUTION_APP_GUIDE.md', bundle / 'README_ja.txt')
    for name in ('README.md', 'README_distribution.md', 'SendLogs.ps1', 'UninstallCleanup.ps1',
                 'puyo_live.example.json', 'mascot_expressions_preview.png'):
        (bundle / name).unlink(missing_ok=True)


def finish_bundle(output: Path, version: str, model: Path, rc_models: Path = RC_MODELS, model_label: str = '24w2-temporary') -> Path:
    bundle = output / 'build/PuyoLive'
    retain_python_vc_runtime(bundle, output)
    write_python_runtime_configs(bundle)
    if digest(bundle / 'app/models/b3/model.joblib') != digest(model):
        raise ValueError('同梱モデルと指定モデルが一致しません')
    for wheel in (output / 'camera_wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(bundle / 'python/Lib/site-packages')
    model_hashes = copy_rc_assets(bundle, rc_models)
    sanitize_bundle_provenance(bundle)
    model_hashes = refresh_model_hashes(bundle, model_hashes)
    copy_release_licenses(bundle)
    prune_release_development_files(bundle)
    metadata = dict(app=version, model=model_label, model_sha256=digest(model), models=model_hashes,
        code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)))
    (bundle / 'app/RELEASE.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for source, dest in (('PuyoLive-distribution.bat', 'PuyoLive.bat'), ('PuyoLive-distribution.bat', 'puyoZeus Live.bat'), ('PuyoLive-distribution.bat', 'puyoZeus.bat')):
        shutil.copy2(ROOT / 'packaging' / source, bundle / dest)
    write_sendlogs(bundle)
    shutil.copy2(ROOT / 'packaging/PuyoLive.vbs', bundle / 'puyoZeus Live.vbs')
    shutil.copy2(ROOT / 'packaging/PuyoLive.vbs', bundle / 'PuyoLive.vbs')
    for name in ('Uninstall.bat', 'アンインストール.bat'):
        shutil.copy2(ROOT / 'packaging/Uninstall.bat', bundle / name)
    shutil.copy2(ROOT / 'packaging/UninstallCleanup.ps1', bundle / 'tools/UninstallCleanup.ps1')
    arrange_launchers(bundle)
    build_launchers(bundle, version)
    (bundle / 'mascot').mkdir(exist_ok=True)
    arrange_release_documents(bundle)
    precompile(bundle / 'python', bundle / 'app', bundle / 'python/Lib/site-packages')
    files = [p for p in (bundle / 'app').rglob('*') if p.is_file() and p.name != 'MANIFEST.json']
    write_manifest(bundle, files)
    return bundle


def artifacts(output: Path, bundle: Path, version: str, zip_only: bool = False) -> dict:
    # インストーラーにも実行状態を持ち込まない。汚れたステージは再構築を要求する。
    if any(is_runtime_file(p.relative_to(bundle).as_posix())
           for p in bundle.rglob('*') if p.is_file()):
        raise ValueError('実行時ファイルがあります。新しいステージで再構築してください')
    release = json.loads((bundle / 'app/RELEASE.json').read_text(encoding='utf-8'))
    if not zip_only:
        command = [str(compiler()), '/Qp', f'/DSourceDir={bundle}', f'/DReleaseDir={output}',
                   f'/DReleaseVersion={version}', f'/DReleaseCodeSHA={release["code_sha"]}',
                   f'/DReleaseModelSHA={release["model_sha256"]}', str(ROOT / 'packaging/PuyoLive.iss')]
        subprocess.run(command, check=True)
    archive = write_bundle_zip(bundle, output / f'puyoZeus-Live-{version}.zip')
    installer = output / f'puyoZeus-Live-{version}-setup.exe'
    result = dict(version=version, files=[dict(name=p.name, sha256=digest(p), size=p.stat().st_size)
                                        for p in ([archive] if zip_only else [installer, archive])])
    (output / 'releases.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--model-label', default='24w2-temporary')
    parser.add_argument('--package-only', action='store_true')
    parser.add_argument('--zip-only', action='store_true')
    parser.add_argument('--rc-models', type=Path, default=RC_MODELS)
    args = parser.parse_args()
    if not args.zip_only:
        compiler()
    output = safe_output(args.output)
    prepare_inputs(output)
    if not args.package_only:
        build(output, DEFAULT_ASSET_ROOT, False, reuse_python=True, with_onnx=True,
              onnx_cache=INPUTS / 'onnx_cache', b3_model=args.model, b3_timing_table=TABLE,
              include_system_vc_runtime=False)
    bundle = finish_bundle(output, args.version, args.model, args.rc_models, args.model_label)
    print(json.dumps(artifacts(output, bundle, args.version, args.zip_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
