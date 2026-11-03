"""充血はコマに焼き込み、速度はWebPのコマ時間だけを変える。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

ASSET_PATTERN = re.compile(r'(overlay|10|30|50|70(?:_low|_high|_normal)?|90(?:_low|_high|_normal)?)(_b[1-3])?(?:_s(0\.5|1|1\.5|2))?\.webp')
FRAME_DURATION_OFFSET = 12
CHUNK_HEADER_SIZE = 8
RIFF_HEADER_SIZE = 12


def retime(data: bytes, speed: float) -> bytes:
    """再圧縮を避ける。画素・透過・ループ情報を一切変更しない。"""
    result = bytearray(data)
    offset = RIFF_HEADER_SIZE
    while offset + CHUNK_HEADER_SIZE <= len(result):
        size = int.from_bytes(result[offset + 4:offset + 8], 'little')
        if result[offset:offset + 4] == b'ANMF':
            position = offset + CHUNK_HEADER_SIZE + FRAME_DURATION_OFFSET
            duration = int.from_bytes(result[position:position + 3], 'little')
            result[position:position + 3] = max(1, round(duration / speed)).to_bytes(3, 'little')
        offset += CHUNK_HEADER_SIZE + size + size % 2
    return bytes(result)


@lru_cache(maxsize=12)
def read_asset(directory: Path, name: str, modified: int) -> bytes:
    match = ASSET_PATTERN.fullmatch(name)
    if match is None:
        raise FileNotFoundError(name)
    stem, bloodshot, speed = match.groups()
    filename = ('overlay_mascot' if stem == 'overlay' else 'mascot_' + stem) + (bloodshot or '') + '.webp'
    data = (directory / filename).read_bytes()
    return retime(data, float(speed)) if speed else data


def mascot_asset(directory: Path, name: str) -> bytes:
    match = ASSET_PATTERN.fullmatch(name)
    if match is None:
        raise FileNotFoundError(name)
    stem, bloodshot, _ = match.groups()
    filename = ('overlay_mascot' if stem == 'overlay' else 'mascot_' + stem) + (bloodshot or '') + '.webp'
    return read_asset(directory, name, (directory / filename).stat().st_mtime_ns)
