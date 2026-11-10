"""署名を保った同梱Pythonの写しを配布の入口にする。"""
from __future__ import annotations
import hashlib
import json
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


def verify_signpath(target: Path, exact_signer: bool = False) -> None:
    """有効な署名と証明書の署名者名の両方を必須にする。"""
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = ("$ErrorActionPreference='Stop';"
              "$s=Get-AuthenticodeSignature -LiteralPath $env:PUYO_SIGNED_COPY;"
              "$name='';if($s.SignerCertificate){"
              "$name=$s.SignerCertificate.GetNameInfo('SimpleName',$false)};"
              "@{status=$s.Status.ToString();signer=$name}|ConvertTo-Json -Compress")
    result = subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', script],
                            env={**os.environ, 'PUYO_SIGNED_COPY': str(target)}, check=True,
                            capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    signature = json.loads(result.stdout)
    signer = signature.get('signer', '')
    accepted = signer == 'SignPath Foundation' if exact_signer else 'SignPath Foundation' in signer
    if signature.get('status') != 'Valid' or not accepted:
        raise ValueError(f'SignPath Foundation の有効な署名がありません: {target}')


def verify_launcher_version(target: Path, version: str) -> None:
    """配布の --version と FileVersion / ProductVersion を一致させる。"""
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = ("$ErrorActionPreference='Stop';"
              "$v=[Diagnostics.FileVersionInfo]::GetVersionInfo($env:PUYO_SIGNED_COPY);"
              "@{file=$v.FileVersion;product=$v.ProductVersion}|ConvertTo-Json -Compress")
    result = subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', script],
                            env={**os.environ, 'PUYO_SIGNED_COPY': str(target)}, check=True,
                            capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    metadata = json.loads(result.stdout)
    if metadata.get('file') != version or metadata.get('product') != version:
        raise ValueError(f'入口の版が配布の版 {version} と一致しません: {target}')


def install_signpath(bundle: Path, source: Path, version: str | None = None) -> None:
    """両方の検証後に入口を配置し、配置したバイトも検証する。"""
    names = ('puyoZeus Live.exe', 'アンインストール.exe')
    for name in names:
        verify_signpath(source / name)
    if version is not None:
        for name in names:
            verify_launcher_version(source / name, version)
    for name in names:
        target = bundle / name
        shutil.copyfile(source / name, target)
        verify_signpath(target)
        if version is not None:
            verify_launcher_version(target, version)
    shutil.copytree(ROOT / 'packaging/launcher', bundle / 'tools/launcher', dirs_exist_ok=True)


def build_launchers(bundle: Path, version: str, signed_launcher_dir: Path | None = None) -> None:
    """元の署名を変更せず、入口にだけsitecustomizeを読み込ませる。"""
    if signed_launcher_dir is not None:
        install_signpath(bundle, signed_launcher_dir, version)
        return
    for name, original in COPIES.items():
        source, target = bundle / 'python' / original, bundle / name
        shutil.copyfile(source, target)
        verify_copy(source, target)
    for name in ('puyoZeus Live', 'アンインストール'):
        (bundle / f'{name}._pth').write_text(PTH, encoding='utf-8')
    shutil.copytree(ROOT / 'packaging/launcher', bundle / 'tools/launcher', dirs_exist_ok=True)
