"""参照amaをLinuxと同系のlibstdc++でWindows AVX2へ交差ビルドする。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess

STAGE = Path(_public_dev_text('<work>/build_20261020a_stage'))
COMPILER = _public_dev_text('<home>/obs_20261020_toolchain/usr/bin/x86_64-w64-mingw32-g++-posix')


def linux_path(value: str) -> str:
    value = value.replace('\\', '/')
    if len(value) > 2 and value[1] == ':':
        return '/mnt/'+value[0].lower()+value[2:]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ama_mingw_r3')
    parser.add_argument('--source-dir',type=Path,default=STAGE/'ama_windows_r2')
    parser.add_argument('--output-root',type=Path,default=STAGE)
    args = parser.parse_args()
    out = args.output_root/args.name
    out.mkdir(exist_ok=False)
    if (args.source_dir/'prepared_sources.json').exists():
        prepared = json.loads((args.source_dir/'prepared_sources.json').read_text())
        previous = ['-I'+p for p in prepared['includes']]+prepared['sources']
    else:
        previous = json.loads((args.source_dir/'command.json').read_text())
    command = [COMPILER, '-O2', '-mavx2', '-mbmi2', '-mpopcnt', '-std=c++14',
               '-DNDEBUG', '-DPUYOPUYO_TSU', '-DNOINLINE_UNLESS_RELEASE=',
               '-DGOOGLE_GLOG_DLL_DECL=', '-DDATA_DIR="data"', '-DTESTDATA_DIR="testdata"',
               '-Wno-attributes', '-ffunction-sections', '-fdata-sections', '-static',
               '-Wl,--gc-sections', _public_dev_text('-I<home>/obs_20261020_toolchain/glog-install/include')]
    for value in previous:
        if 'third_party' in value:
            continue
        if value.startswith('-I'):
            command.append('-I'+linux_path(value[2:]))
        elif value.endswith(('.cc', '.cpp')) and not value.endswith(('flags.cc', 'kumipuyo_seq_generator.cc')):
            command.append(linux_path(value))
    command += [_public_dev_text('<home>/obs_20261020_toolchain/glog-install/lib/libglog.a'), '-ldbghelp',
                '-include', str(args.source_dir/'source/build/build_config.h'),
                '-o', str(out/'ama_reference.exe')]
    env = dict(os.environ, PATH=_public_dev_text('<home>/obs_20261020_toolchain/usr/bin:')+os.environ['PATH'])
    (out/'command.json').write_text(json.dumps(command, indent=2))
    with (out/'build.log').open('w') as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt = dict(returncode=done.returncode, affinity=sorted(os.sched_getaffinity(0)),
                   nice=os.getpriority(os.PRIO_PROCESS, 0))
    if done.returncode == 0:
        receipt['binary_sha256'] = hashlib.sha256((out/'ama_reference.exe').read_bytes()).hexdigest()
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    main()
