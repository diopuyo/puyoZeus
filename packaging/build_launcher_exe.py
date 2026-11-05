"""標準の .NET Framework csc で署名前の入口を再現可能に組み立てる。"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('puyoZeus Live.exe', 'アンインストール.exe')
PE_POINTER = 0x3C
PE_TIMESTAMP_OFFSET = 8
SECTION_SIZE = 40
CLI_DIRECTORY_OFFSET = 208  # PE32 optional header: data directory 14
ALIGNMENT = 4


def compiler() -> Path:
    """Windows に標準搭載された Framework コンパイラを探す。"""
    windows = Path(os.environ.get('SystemRoot', 'C:/Windows'))
    for framework in ('Framework64', 'Framework'):
        path = windows / 'Microsoft.NET' / framework / 'v4.0.30319/csc.exe'
        if path.is_file():
            return path
    raise FileNotFoundError('.NET Framework 4.x csc.exe がありません')


def metadata_offset(data: bytearray, pe: int) -> int:
    """PE32 の CLI メタデータをセクションの RVA から求める。"""
    count = struct.unpack_from('<H', data, pe + 6)[0]
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from('<H', data, optional)[0] != 0x10B:
        raise ValueError('PE32 以外は未対応')
    sections = optional + optional_size

    def offset(rva: int) -> int:
        for index in range(count):
            pos = sections + index * SECTION_SIZE
            size, start, raw_size, raw = struct.unpack_from('<IIII', data, pos + 8)
            if start <= rva < start + max(size, raw_size):
                return raw + rva - start
        raise ValueError('不正な RVA')

    cli = offset(struct.unpack_from('<I', data, optional + CLI_DIRECTORY_OFFSET)[0])
    return offset(struct.unpack_from('<I', data, cli + 8)[0])


def normalize(path: Path) -> None:
    """旧 csc の時刻と MVID を内容由来にする。必ず署名前だけに適用する。"""
    data = bytearray(path.read_bytes())
    pe = struct.unpack_from('<I', data, PE_POINTER)[0]
    struct.pack_into('<I', data, pe + PE_TIMESTAMP_OFFSET, 0)
    metadata = metadata_offset(data, pe)
    version_size = struct.unpack_from('<I', data, metadata + 12)[0]
    pos = metadata + 16 + version_size
    streams = struct.unpack_from('<H', data, pos + 2)[0]
    pos += 4
    for _ in range(streams):
        offset, size = struct.unpack_from('<II', data, pos)
        end = data.index(0, pos + 8)
        name = bytes(data[pos + 8:end])
        pos = (end + ALIGNMENT) // ALIGNMENT * ALIGNMENT
        if name == b'#GUID':
            if size != 16:
                raise ValueError('MVID 以外の GUID があるため正規化を中止')
            begin = metadata + offset
            data[begin:begin + size] = bytes(size)
            data[begin:begin + size] = hashlib.sha256(data).digest()[:size]
            path.write_bytes(data)
            return
    raise ValueError('MVID がありません')


def build(output: Path, version: str, csc: Path | None = None) -> None:
    """同一ソース・版・コンパイラなら出力先によらず同じバイトにする。"""
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:\.\d+)?', version):
        raise ValueError('版は major.minor.patch[.revision] で指定してください')
    if any(int(part) > 65534 for part in version.split('.')):
        raise ValueError('版の各数値は 65534 以下で指定してください')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in NAMES):
        raise FileExistsError('既存の exe は上書きしません')
    attributes = ('using System.Reflection;\n'
                  '[assembly: AssemblyTitle("puyoZeus Live")]\n'
                  '[assembly: AssemblyProduct("puyoZeus Live")]\n'
                  '[assembly: AssemblyCopyright("Copyright puyoZeus contributors. MIT License")]\n'
                  f'[assembly: AssemblyVersion("{version}")]\n'
                  f'[assembly: AssemblyFileVersion("{version}")]\n'
                  f'[assembly: AssemblyInformationalVersion("{version}")]\n')
    with tempfile.TemporaryDirectory(dir=output) as directory:
        info = Path(directory) / 'AssemblyInfo.cs'
        info.write_text(attributes, encoding='utf-8')
        for uninstall, name in enumerate(NAMES):
            target = output / name
            command = [str(csc or compiler()), '/nologo', '/optimize+', '/debug-',
                       '/platform:anycpu', '/target:exe' if uninstall else '/target:winexe',
                       '/reference:System.Windows.Forms.dll', f'/out:{target}',
                       f'/win32icon:{ROOT / "src/phase_j/brand/app_logo.ico"}']
            if uninstall:
                command.append('/define:UNINSTALL')
            subprocess.run([*command, str(ROOT / 'packaging/launcher_exe/Launcher.cs'),
                            str(info)], check=True)
            normalize(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.version)


if __name__ == '__main__':
    main()
