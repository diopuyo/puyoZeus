"""Windows上でcp312計算部品と単盤面DLLを固定ツールで組む（署名要求は別段）。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from native_version import stamp_version, version_blob

ROOT = Path(__file__).resolve().parents[1]
RUST_VERSION = '1.97.1'
MATURIN_VERSION = '1.9.4'
ZIG_VERSION = '0.16.0'
TARGET = 'x86_64-pc-windows-gnu'
FEATURES = ('ama-fast-avx2,ama-fast-connected,ama-fast-gravity,'
            'ama-reuse-identical-queues,b3-fast-prefilter,b3-fast-chain,compact-chain-summary')
PYD = 'puyo_core.cp312-win_amd64.pyd'
DLL = 'puyo_live_exact.dll'


def build(output: Path, work: Path, version: str) -> None:
    """既存の成果物を上書きせず、すべて新しい作業先で組み立てる。"""
    version_blob(version)
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 12):
        raise ValueError('WindowsのPython 3.12で実行してください')
    rust = subprocess.check_output(['rustc', '--version'], text=True).split()[1]
    maturin = subprocess.check_output([sys.executable, '-m', 'maturin', '--version'], text=True).split()[1]
    zig = subprocess.check_output([sys.executable, '-m', 'ziglang', 'version'], text=True).strip()
    if (rust, maturin, zig) != (RUST_VERSION, MATURIN_VERSION, ZIG_VERSION):
        raise ValueError(f'固定版が必要です: Rust {RUST_VERSION}, maturin {MATURIN_VERSION}, Zig {ZIG_VERSION}')
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    env = build_environment(work)
    subprocess.run([sys.executable, '-m', 'maturin', 'build', '--locked', '--release',
                    '--target', TARGET, '--interpreter', sys.executable, '--features', FEATURES,
                    '--out', str(work / 'wheels')], cwd=ROOT / 'native/puyo_core', env=env, check=True)
    wheels = list((work / 'wheels').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError(f'wheel件数が不正: {len(wheels)}/1')
    with zipfile.ZipFile(wheels[0]) as archive:
        members = [name for name in archive.namelist() if name.endswith('/' + PYD)]
        if len(members) != 1:
            raise ValueError(f'cp312 pyd件数が不正: {len(members)}/1')
        (output / PYD).write_bytes(archive.read(members[0]))
    subprocess.run(['rustc', '--edition=2021', '--crate-type=cdylib', '--target', TARGET,
                    '-C', 'opt-level=3', '-C', 'panic=abort',
                    '-C', f'linker={work / "zig-link.cmd"}',
                    '--remap-path-prefix', f'{ROOT}=build-source',
                    '--remap-path-prefix', f'{Path.home()}=build-home',
                    str(ROOT / 'native/puyo_core/src/live_exact.rs'), '-o', str(output / DLL)],
                   env=env, check=True)
    for name in (PYD, DLL):
        stamp_version(output / name, version)


def build_environment(work: Path) -> dict[str, str]:
    """個人パスとランナーの配置場所をコンパイル結果から除く。"""
    flags = ['-C', 'target-cpu=x86-64',
             '--remap-path-prefix', f'{ROOT}=build-source',
             '--remap-path-prefix', f'{Path.home()}=build-home',
             '--remap-path-prefix', f'{work}=build-output']
    for name in ('CARGO_HOME', 'RUSTUP_HOME'):
        if os.environ.get(name):
            flags.extend(['--remap-path-prefix', f'{os.environ[name]}=build-tools/{name.lower()}'])
    env = {key: value for key, value in os.environ.items() if key not in
           ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'PYO3_CONFIG_FILE')}
    env.update(CARGO_TARGET_DIR=str(work / 'target'), CARGO_ENCODED_RUSTFLAGS='\x1f'.join(flags),
               CARGO_BUILD_JOBS='4', PYO3_PYTHON=sys.executable,
               ZIG_GLOBAL_CACHE_DIR=str(work / 'zig-cache'),
               CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=str(work / 'zig-link.cmd'))
    command = f'@"{sys.executable}" "{ROOT / "packaging/native_zig_link.py"}" %*\n'
    (work / 'zig-link.cmd').write_text(command, encoding='utf-8')
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.work.resolve(), args.version)


if __name__ == '__main__':
    main()
