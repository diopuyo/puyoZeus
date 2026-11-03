"""旧配布を保存し、同じ収集規則で本人用b版を新規作成する。"""
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
import shutil
import sys
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from bundle_archive import extract_smoke_copy, write_bundle_zip, is_runtime_file
from build_bundle import collect_app_files, source_of, precompile, write_manifest
from build_live_obs_20261020 import settings, ASSETS, sha
from seal_live_v49f import inventory
import make_live_obs_launchers as launchers

WORK = Path('D:/puyo_analyzer/packaging/build_20261020b_work')
BUILD = 'build_20261020c'
OUT = Path('D:/puyo_analyzer/packaging')/(BUILD+'_release')
BUNDLE = WORK/'stage/PuyoLive'
OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261020a_release/PuyoLive_20261020a_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
ENTRY = Path('D:/PuyoLive_OBS_20261020c_USER_ONLY')
FAST = Path(_public_dev_text('<work>/build_20261020c_validation/PuyoLive'))
PRELOAD_ARCHIVES = False
PRECOMPILE_RUNTIME = None
IO_WORKERS = 32


def open_zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(BytesIO(path.read_bytes()) if PRELOAD_ARCHIVES else path)


def break_link(path: Path, keep: bool = True) -> None:
    """共有資産を書き換える前に、この作業コピーだけを独立させる。"""
    if path.is_file() and path.stat().st_nlink > 1:
        data = path.read_bytes() if keep else None
        path.unlink()
        if data is not None:
            path.write_bytes(data)


def stage() -> None:
    extract_smoke_copy(OLD_ZIP, BUNDLE.parent)
    OUT.mkdir(exist_ok=True)
    refresh()


def refresh() -> None:
    assert not (OUT/'archive_receipt.json').exists()
    OUT.mkdir(exist_ok=True)
    for path in BUNDLE.iterdir():
        if path.is_file():
            break_link(path)
    # fastコピーではapp/python以外の同梱文書も元ZIPから必ず復元する。
    with open_zip(OLD_ZIP) as archive:
        licenses = [(BUNDLE/name.removeprefix('PuyoLive/'), archive.read(name))
                    for name in archive.namelist()
                    if name.startswith('PuyoLive/LICENSES/') and not name.endswith('/')]
    with ThreadPoolExecutor(max_workers=IO_WORKERS if PRELOAD_ARCHIVES else 1) as workers:
        list(workers.map(restore_license, licenses))
    relatives = collect_app_files(ASSETS/'baseline_b/app')
    with ThreadPoolExecutor(max_workers=IO_WORKERS if PRELOAD_ARCHIVES else 1) as workers:
        changed = [p for p in workers.map(refresh_source, relatives) if p is not None]
    compile_targets = changed_python(relatives) if PRELOAD_ARCHIVES else [p for p in changed if p.suffix == '.py']
    precompile(PRECOMPILE_RUNTIME or BUNDLE/'python', *compile_targets) if compile_targets else None
    values = dict(settings(), live_score_cadence=True, live_static_exact=True)
    (BUNDLE/'puyo_live.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    original_guide = launchers.guide
    launchers.guide = lambda *a, **k: original_guide(*a, **k).replace('build_20261020a', BUILD).replace(
        '試験前にPC1の重い処理を止めてください。', '最優先の盤面認識の根本対策は止めず、空いている時間に試験してください。')
    launchers.populate(BUNDLE, Path('%~dp0.'), False)
    launchers.cmd(BUNDLE/'PuyoLive.bat', ['@echo off', 'call "%~dp0Start_PuyoLive_USER_ONLY.cmd"'])
    break_link(BUNDLE/'app/MANIFEST.json', keep=False)
    manifest_files = declared_app_files(relatives, compile_targets) if PRELOAD_ARCHIVES else [
        p for p in (BUNDLE/'app').rglob('*') if p.is_file() and p.name != 'MANIFEST.json'
        and not is_runtime_file(p.relative_to(BUNDLE).as_posix())]
    (parallel_manifest if PRELOAD_ARCHIVES else write_manifest)(BUNDLE, manifest_files)
    (BUNDLE.parent/'stage_receipt.json').write_text(json.dumps(dict(changed=[str(p.relative_to(BUNDLE)) for p in changed],
        settings=values, old_zip_sha256=sha(OLD_ZIP)), indent=2), encoding='utf-8')
    print('Stage ready', BUNDLE, flush=True)


def refresh_source(relative: str) -> Path | None:
    source, target = source_of(relative, ASSETS/'baseline_b/app'), BUNDLE/'app'/relative
    if target.exists() and source.read_bytes() == target.read_bytes():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    break_link(target, keep=False)
    shutil.copy2(source, target)
    if target.suffix == '.py':
        break_link(target.parent/'__pycache__'/f'{target.stem}.cpython-312.pyc', keep=False)
    return target


def restore_license(item: tuple[Path, bytes]) -> None:
    target, payload = item
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != payload:
        break_link(target, keep=False)
        target.write_bytes(payload)


def changed_python(relatives: list[str]) -> list[Path]:
    """中断後にsourceだけコピー済みでも、旧ZIPとの差分pycは必ず作り直す。"""
    targets = []
    with open_zip(OLD_ZIP) as archive:
        names = set(archive.namelist())
        for relative in relatives:
            if not relative.endswith('.py'):
                continue
            name = 'PuyoLive/app/'+relative
            if name not in names or source_of(relative, ASSETS/'baseline_b/app').read_bytes() != archive.read(name):
                targets.append(BUNDLE/'app'/relative)
    return targets


def declared_app_files(relatives: list[str], changed: list[Path]) -> list[Path]:
    """完成元と同じ収集規則の追加分を列挙し、一件ずつのSMB statを避ける。"""
    with open_zip(OLD_ZIP) as archive:
        names = {i.filename.removeprefix('PuyoLive/app/') for i in archive.infolist()
                 if i.filename.startswith('PuyoLive/app/') and not i.is_dir()}
    names.update(relatives)
    names.update((p.parent/'__pycache__'/f'{p.stem}.cpython-312.pyc').relative_to(BUNDLE/'app').as_posix()
                 for p in changed if p.suffix == '.py')
    return [BUNDLE/'app'/name for name in sorted(names)
            if name != 'MANIFEST.json' and not is_runtime_file('app/'+name)]


def manifest_entry(path: Path) -> tuple[str, dict]:
    payload = path.read_bytes()
    return path.relative_to(BUNDLE/'app').as_posix(), dict(size=len(payload), sha256=sha256(payload).hexdigest())


def parallel_manifest(bundle: Path, files: list[Path]) -> None:
    """生成時も全ファイルを読み、ネットワークI/Oだけを並行化する。"""
    from src.phase_j.launcher_manifest import installed_versions, MANIFEST_SCHEMA
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as workers:
        entries = dict(workers.map(manifest_entry, sorted(set(files))))
    packages = installed_versions(bundle/'python/Lib/site-packages')
    value = dict(schema=MANIFEST_SCHEMA, files=entries, packages=dict(sorted(packages.items())))
    (bundle/'app/MANIFEST.json').write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def packaged_files() -> list[str]:
    """完成元の一覧と新MANIFESTを使い、SMB上のruntime全ディレクトリ探索を避ける。"""
    with open_zip(OLD_ZIP) as archive:
        names = {i.filename.removeprefix('PuyoLive/') for i in archive.infolist() if not i.is_dir()}
    manifest = json.loads((BUNDLE/'app/MANIFEST.json').read_text(encoding='utf-8'))
    names.update('app/'+name for name in manifest['files'])
    names.update(p.name for p in BUNDLE.iterdir() if p.is_file())
    return sorted(names)


def fast_stage() -> None:
    """C:の検証コピー。Pythonだけ不変ファイルをhardlinkし、appは独立コピーする。"""
    global BUNDLE
    BUNDLE = FAST
    old = Path(_public_dev_text('<work>/build_20261020a_stage/zip_smoke_copy_fast/PuyoLive'))
    assert not BUNDLE.exists()
    BUNDLE.mkdir(parents=True)
    shutil.copytree(old/'python', BUNDLE/'python', copy_function=os.link)
    # runtimeのファイルには書かない。unchecked-hashのapp pycは独立コピー後に再生成する。
    shutil.copytree(old/'app', BUNDLE/'app', ignore=shutil.ignore_patterns('device_calibration'))
    for path in old.iterdir():
        if path.is_file():
            shutil.copy2(path, BUNDLE/path.name)
    OUT.mkdir(exist_ok=True)
    refresh()


def seal() -> None:
    sources: dict[str, str] = {}
    options = dict(relative_files=packaged_files(), source_hashes=sources) if PRELOAD_ARCHIVES else {}
    archive = write_bundle_zip(BUNDLE, OUT/f'PuyoLive_{BUILD.removeprefix("build_")}_USER_ONLY_DO_NOT_DISTRIBUTE.zip', **options)
    after, before = inventory(archive), inventory(OLD_ZIP)
    assert not [p for p in after if is_runtime_file(p.removeprefix('PuyoLive/'))]
    changes = dict(added=sorted(after.keys()-before.keys()), removed=sorted(before.keys()-after.keys()),
                   changed=[p for p in sorted(after.keys() & before.keys()) if after[p] != before[p]])
    assert not changes['removed'], changes['removed']
    assert all(after[p] == before[p] for p in before if p.startswith('PuyoLive/LICENSES/'))
    receipt = dict(zip=str(archive), sha256=sha(archive), files=len(after), changes=changes)
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (OUT/'files_sha256.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
    (OUT/'SHA256SUMS.txt').write_text(f'{receipt["sha256"]}  {archive.name}\n', encoding='ascii')
    # 全ZIP memberのbytes一致を確認した検証コピーを起動先にする。
    with open_zip(archive) as source:
        assert source.testzip() is None
        manifest = json.loads(source.read('PuyoLive/app/MANIFEST.json'))
        for name, value in manifest['files'].items():
            assert after['PuyoLive/app/'+name] == value['sha256'], name
        if PRELOAD_ARCHIVES:
            assert after == sources, 'ZIP再読込みの全hashと、組立時に読んだ全bytesのhashが一致しない'
        else:
            for name in source.namelist():
                if not name.endswith('/'):
                    assert source.read(name) == (BUNDLE/name.removeprefix('PuyoLive/')).read_bytes(), name
    receipt['smoke_copy'] = str(BUNDLE)
    receipt['all_archive_members_byte_equal'] = True
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    launchers.BUNDLE = BUNDLE
    launchers.populate(ENTRY, launchers.BUNDLE, True)
    guide = ENTRY/'OBS_TEST_README_ja.txt'
    guide.write_text(launchers.guide(BUNDLE, True).replace('build_20261020a', BUILD).replace(
        '試験前にPC1の重い処理を止めてください。', '最優先の盤面認識の根本対策は止めず、空いている時間に試験してください。'), encoding='utf-8-sig')
    print(json.dumps(dict(zip=str(archive), sha256=receipt['sha256'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'refresh', 'seal', 'fast_stage'))
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    if args.fast:
        BUNDLE = FAST
    globals()[args.action]()
