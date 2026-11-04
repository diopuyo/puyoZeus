"""署名を保った同梱Pythonの写しを配布の入口にする。"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COPIES = {'puyoZeus Live.exe': 'pythonw.exe', 'アンインストール.exe': 'python.exe',
          'python312.dll': 'python312.dll', 'vcruntime140.dll': 'vcruntime140.dll',
          'vcruntime140_1.dll': 'vcruntime140_1.dll'}
PTH = 'python\\python312.zip\npython\ntools\\launcher\nimport site\n'


def verify_copy(source: Path, target: Path) -> None:
    """バイト一致と写しの有効な署名を必須にする。"""
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
        raise ValueError(f'入口の写しが不一致: {target}')
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = "$s=Get-AuthenticodeSignature -LiteralPath $env:PUYO_SIGNED_COPY; if($s.Status -ne 'Valid'){exit 1}"
    subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', script],
                   env={**os.environ, 'PUYO_SIGNED_COPY': str(target)}, check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def build_launchers(bundle: Path, version: str) -> None:
    """元の署名を変更せず、入口にだけsitecustomizeを読み込ませる。"""
    for name, original in COPIES.items():
        source, target = bundle / 'python' / original, bundle / name
        shutil.copyfile(source, target)
        verify_copy(source, target)
    for name in ('puyoZeus Live', 'アンインストール'):
        (bundle / f'{name}._pth').write_text(PTH, encoding='utf-8')
    shutil.copytree(ROOT / 'packaging/launcher', bundle / 'tools/launcher', dirs_exist_ok=True)
