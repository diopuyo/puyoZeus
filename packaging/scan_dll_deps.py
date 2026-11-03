"""配布物 python/ 配下の全 DLL/.pyd/.exe が import する DLL を PE ヘッダから走査する。

    python packaging/scan_dll_deps.py <PuyoLive/python> [--pefile-path <pefile の在る場所>]

対象 PC に VC++ 再頒布可能パッケージが無くても、「何が要るか」をこの走査結果で示すための道具
(動かして確かめるのはその PC でしか出来ない)。分類:
  bundled   : 配布物内に同名ファイルがある (追加不要)
  apiset    : api-ms-win-* / ext-ms-* (Windows 10+ の OS 部品。UCRT は OS 標準)
  vc_runtime: VC++ ランタイム (vcruntime140* / msvcp140* / concrt140 / vcomp140)
  system    : 上記以外で、この PC の System32 に在る Windows 標準 DLL
  unknown   : どれでもない (要調査)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

VC_PREFIXES = ('vcruntime140', 'msvcp140', 'concrt140', 'vcomp140', 'vcamp140')
APISET_PREFIXES = ('api-ms-win-', 'ext-ms-')
SUFFIXES = ('.dll', '.pyd', '.exe')
SYSTEM_DIR = Path('C:/Windows/System32')
EXAMPLES = 3
SITE_PACKAGES = Path('Lib/site-packages')


def classify(name: str, bundled: set[str], system: set[str]) -> str:
    if name in bundled:
        return 'bundled'
    if name.startswith(APISET_PREFIXES):
        return 'apiset'
    if name.startswith(VC_PREFIXES):
        return 'vc_runtime'
    return 'system' if name in system else 'unknown'


def imports_of(path: Path, pefile: object) -> set[str]:
    """通常 import と delay-load import の DLL 名 (小文字)。"""
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT']])
    names: set[str] = set()
    for attribute in ('DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT'):
        names |= {entry.dll.decode('ascii', 'replace').lower() for entry in getattr(pe, attribute, [])}
    pe.close()
    return names


def local_dirs(path: Path, root: Path) -> list[Path]:
    """importer が DLL を解決できる配布物内ディレクトリ (実ローダの探索を近似)。
    自分のディレクトリ / python.exe のディレクトリ (アプリ ディレクトリ) / パッケージ同梱の libs
    (delvewheel の `<pkg>.libs`、`<pkg>/.libs`、torch の `<pkg>/lib`)。"""
    dirs = [path.parent, root]
    site = root / SITE_PACKAGES
    try:
        top = path.relative_to(site).parts[0]
    except ValueError:
        return dirs
    return dirs + [site / f'{top}.libs', site / top / '.libs', site / top / 'lib']


def scan(root: Path, pefile: object) -> dict:
    """importer ごとに解決先を判定する。同名が別のパッケージ内にあるだけでは解決済みとしない
    (実例: msvcp140.dll は sklearn/.libs にしか無く、torch からは見えない)。"""
    files = [p for p in root.rglob('*') if p.suffix.lower() in SUFFIXES and p.is_file()]
    system = {p.name.lower() for p in SYSTEM_DIR.glob('*.dll')} if SYSTEM_DIR.is_dir() else set()
    users: dict[str, dict[str, list[str]]] = {}
    for path in files:
        for name in imports_of(path, pefile):
            local = any((d / name).exists() or any(x.name.lower() == name for x in d.glob('*.dll'))
                        for d in local_dirs(path, root) if d.is_dir())
            kind = 'bundled' if local else classify(name, set(), system)
            users.setdefault(kind, {}).setdefault(name, []).append(path.relative_to(root).as_posix())
    groups = {kind: {name: dict(importers=len(v), examples=v[:EXAMPLES]) for name, v in sorted(names.items())}
              for kind, names in users.items()}
    distinct = len({n for names in users.values() for n in names})
    return dict(files_scanned=len(files), distinct_imports=distinct, groups=groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('python_dir', type=Path)
    parser.add_argument('--pefile-path', type=Path, help='pip install --target した pefile の場所')
    options = parser.parse_args()
    if options.pefile_path:
        sys.path.insert(0, str(options.pefile_path))
    import pefile
    print(json.dumps(scan(options.python_dir, pefile), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
