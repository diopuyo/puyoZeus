"""本人用OBS試験のama参照C++をWindows AVX2へビルドする。原本は変更しない。"""
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
import re
import shutil
import subprocess
import tarfile
import urllib.request

REFERENCE = Path(_public_dev_text('<work>/ama_reference_profile_20261016/puyoai-private-main'))
PLAIN = Path(_public_dev_text('<work>/wt_ama_plain'))
ZIG = Path('D:/puyo_analyzer/packaging/native_build/zx/ziglang/zig.exe')
JSON_URL = 'https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.5.tar.gz'


def prepare(out: Path) -> tuple[list[Path], list[Path]]:
    """参照と研究版の修正済み実装を専用コピーへ収録する。"""
    src = out/'source'
    if not src.exists():
        shutil.copytree(REFERENCE/'src', src)
    for name in ('cpu/ama_reference/placement.cc', 'core/bit_field.cc'):
        shutil.copy2(PLAIN/'logs/ama_plain/patched'/name, src/name)
    shutil.copy2(PLAIN/'logs/ama_plain/search_serial.cc', src/'cpu/ama_reference/beam/search.cc')
    for name in ('ama_plain_reference.cpp', 'ama_plain_pathsearch.cc', 'ama_plain_pathsearch.h'):
        shutil.copy2(PLAIN/'scripts'/name, src/name)
    archive = out/'jsoncpp-1.9.5.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(JSON_URL, archive)
    if not (out/'jsoncpp-1.9.5').exists():
        with tarfile.open(archive) as stream:
            stream.extractall(out, filter='data')
    jsrc = out/'jsoncpp-1.9.5'
    wrapper = out/'include/jsoncpp/json'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    if not wrapper.exists():
        shutil.copytree(jsrc/'include/json', wrapper)
    glog = REFERENCE/'third_party/glog/glog/src'
    sources = [src/'ama_plain_reference.cpp', src/'ama_plain_pathsearch.cc']
    for target in ('src/core/core', 'src/cpu/ama_reference/beam_core',
                   'src/cpu/ama_reference/eval_core', 'src/cpu/fold_form_db/fold_form_db'):
        ninja = (REFERENCE/f'speed_build/obj/{target}.ninja').read_text()
        sources.extend(src/p for p in re.findall(r': cxx ../src/(\S+)', ninja))
    sources.extend(src/p for p in ('base/strings.cc', 'base/file/file.cc', 'base/file/path.cc'))
    sources.extend(jsrc/'src/lib_json'/p for p in ('json_reader.cpp', 'json_value.cpp', 'json_writer.cpp'))
    sources.extend(glog/p for p in ('logging.cc', 'raw_logging.cc', 'utilities.cc', 'vlog_is_on.cc', 'windows/port.cc'))
    return sources, [src, out/'include', jsrc/'include', glog/'windows', glog]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources, includes = prepare(out)
    if args.prepare_only:
        (out/'prepared_sources.json').write_text(json.dumps(dict(sources=list(map(str,sources)),
            includes=list(map(str,includes))),indent=2),encoding='utf-8')
        return
    local_zig = out/'zig.exe'
    if not local_zig.exists():
        shutil.copy2(ZIG, local_zig)
    os.environ['ZIG_LIB_DIR'] = str(ZIG.parent/'lib')
    command = [str(local_zig), 'c++', '-target', 'x86_64-windows-gnu', '-O2', '-mavx2',
               '-mbmi2', '-mpopcnt', '-std=c++14', '-DNDEBUG', '-DPUYOPUYO_TSU',
               '-DNOINLINE_UNLESS_RELEASE=', '-DGOOGLE_GLOG_DLL_DECL=',
               '-DDATA_DIR="data"', '-DTESTDATA_DIR="testdata"',
               '-include', str(out/'source/build/build_config.h')]
    command.extend('-I'+str(p) for p in includes)
    command.extend(str(p) for p in sources)
    command.extend(['-o', str(out/'ama_reference.exe')])
    (out/'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    with (out/'build.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = dict(returncode=result.returncode, avx2=True,
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if result.returncode == 0:
        receipt['binary_sha256'] = hashlib.sha256((out/'ama_reference.exe').read_bytes()).hexdigest()
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
