"""明示指定された計算部品だけを、署名・版・計算結果の照合後に同梱する。"""
from __future__ import annotations

import ctypes
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType

from build_launchers import verify_launcher_version, verify_signpath

ROOT = Path(__file__).resolve().parents[1]
PYD = 'puyo_core.cp312-win_amd64.pyd'
DLL = 'puyo_live_exact.dll'
DESTINATIONS = {PYD: 'python/Lib/site-packages/puyo_core/' + PYD,
                DLL: 'app/src/phase_j/' + DLL}
CELLS = 78
RECORD = CELLS * 3
MAX_STEPS = CELLS // 4
SAMPLES = 240
SEED = 20261013
TIMEOUT_SECONDS = 120


def dll_fingerprint(path: Path) -> str:
    """全C ABIを、隠し段・穴・UNKNOWN・おじゃまを含む固定入力で照合する。"""
    library = ctypes.CDLL(str(path.resolve()))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    library.live_stable.argtypes = [pointer, ctypes.c_size_t]
    library.live_stable.restype = ctypes.c_bool
    library.live_local.argtypes = [pointer, pointer, ctypes.c_size_t, ctypes.c_size_t]
    library.live_local.restype = ctypes.c_bool
    library.live_simulate.argtypes = [pointer, ctypes.c_size_t, pointer]
    library.live_simulate.restype = ctypes.c_size_t
    rng = random.Random(SEED)
    digest = hashlib.sha256()
    for index in range(SAMPLES):
        values = [rng.choice([0, 1, 2, 3, 4, 5, 9, 10]) for _ in range(CELLS)]
        if index < 6:
            values = [0] * CELLS
            values[-4:] = [index] * 4
        for hidden in (0, 1):
            grid = (ctypes.c_uint8 * CELLS)(*values)
            starts = (ctypes.c_uint8 * CELLS)(*range(CELLS))
            output = (ctypes.c_uint8 * (MAX_STEPS * RECORD))()
            stable = library.live_stable(grid, hidden)
            local = library.live_local(grid, starts, CELLS, hidden)
            count = library.live_simulate(grid, hidden, output)
            if count > MAX_STEPS:
                raise ValueError('単盤面DLLの連鎖数が範囲外です')
            digest.update(bytes([stable, local, count]))
            digest.update(bytes(output[:count * RECORD]))
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """計算の照合だけに必要なファイルを読み、アプリ全体の初期化を避ける。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'計算の照合ファイルを読み込めません: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(source: Path, reference: Path | None = None) -> dict:
    """保存済み指紋と比較する。旧DLLの実行を要求せずSACの影響を避ける。"""
    golden = json.loads(Path(__file__).with_name('native_golden.json').read_text('utf-8'))
    if (golden['format'], golden['seed'], golden['samples'], golden['hidden_rows']) != (1, SEED, SAMPLES, [0, 1]):
        raise ValueError('DLL指紋の入力条件が不一致です')
    if reference is not None and hashlib.sha256(reference.read_bytes()).hexdigest() != golden['reference_dll_sha256']:
        raise ValueError('基準DLLのSHA256が不一致です')
    checks = load_module('native_fingerprint_reference', ROOT / 'src/phase_j/live_ex_native.py')
    native = load_module('puyo_core', source / PYD)
    verified = checks.verify_native(native)
    expected = golden['dll_fingerprint']
    actual = dll_fingerprint(source / DLL)
    if actual != expected:
        raise ValueError('単盤面DLLの指紋が不一致: 1/1 (240盤面×隠し段2条件)')
    return dict(pyd_fingerprints=len(verified['fingerprint']), dll_cases=SAMPLES * 2,
                dll_fingerprint=actual)


def verify_contents(source: Path) -> None:
    """読み込み不能・不一致・時間超過はすべて採用せず停止する。"""
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), str(source)], check=True,
                   capture_output=True, text=True, timeout=TIMEOUT_SECONDS)


def verify_signed_files(source: Path, version: str) -> None:
    """両ファイルの署名と版を確認する。CIの署名済み成果物にも同じ門を使う。"""
    for name in DESTINATIONS:
        verify_signpath(source / name, exact_signer=True)
        verify_launcher_version(source / name, version)


def install_signed_native(bundle: Path, source: Path | None, version: str) -> None:
    """既定は何も書き換えない。検査対象と実際に同梱するバイトを揃える。"""
    if source is None:
        return
    with tempfile.TemporaryDirectory(prefix='signed-native-') as temporary:
        staged = Path(temporary)
        for name, relative in DESTINATIONS.items():
            if not (bundle / relative).is_file():
                raise ValueError(f'計算部品の置換先がありません: {relative}')
            shutil.copyfile(source / name, staged / name)
        verify_signed_files(staged, version)
        verify_contents(staged)
        for name, relative in DESTINATIONS.items():
            shutil.copyfile(staged / name, bundle / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path, nargs='?')
    parser.add_argument('--signed-version')
    args = parser.parse_args()
    if args.signed_version is not None:
        verify_signed_files(args.source, args.signed_version)
    print(json.dumps(probe(args.source, args.reference), sort_keys=True))


if __name__ == '__main__':
    main()
