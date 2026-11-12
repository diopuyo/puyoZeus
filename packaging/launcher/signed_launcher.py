"""標準ライブラリだけで署名付き入口の確認と起動を行う。"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Any
import urllib.request

VC_URL = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
WEB_URL = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'
STARTUP_SECONDS = 8
DOWNLOAD_SECONDS = 120
SUCCESS_CODES = (0, 3010, 1641)
YES = 6
LANG_JAPANESE = 0x11
PRIMARY_LANGUAGE_MASK = 0x3ff
MESSAGE_QUESTION = 0x24
MESSAGE_WARNING = 0x30
SEE_MASK_NOCLOSEPROCESS = 0x40
INFINITE = 0xffffffff
SW_SHOWNORMAL = 1
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80


def japanese() -> bool:
    return ctypes.windll.kernel32.GetUserDefaultUILanguage() & PRIMARY_LANGUAGE_MASK == LANG_JAPANESE


def text(ja: str, en: str) -> str:
    return ja if japanese() else en


def message(value: str, question: bool = False) -> bool:
    return ctypes.windll.user32.MessageBoxW(None, value, 'puyoZeus Live',
        MESSAGE_QUESTION if question else MESSAGE_WARNING) == YES


def has_vc() -> bool:
    system = Path(os.environ['SystemRoot']) / 'System32'
    return all((system / name).is_file() for name in ('msvcp140.dll', 'vcruntime140_1.dll'))


def has_webview() -> bool:
    import winreg
    key = r'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as entry:
                    version = str(winreg.QueryValueEx(entry, 'pv')[0])
                    if all(part.isdigit() for part in version.split('.')) and any(
                            int(part) > 0 for part in version.split('.')):
                        return True
            except OSError:
                pass
    return False


class HttpsRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        if not newurl.lower().startswith('https://'):
            raise ValueError('HTTPS required')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, target: Path) -> None:
    if url not in (VC_URL, WEB_URL):
        raise ValueError('Unexpected installer URL')
    opener = urllib.request.build_opener(HttpsRedirect())
    with opener.open(url, timeout=DOWNLOAD_SECONDS) as response, target.open('xb') as stream:
        if not response.url.lower().startswith('https://'):
            raise ValueError('HTTPS required')
        shutil.copyfileobj(response, stream)


def verify_microsoft(target: Path) -> bool:
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = ("$s=Get-AuthenticodeSignature -LiteralPath $env:PUYO_INSTALLER;"
              "if($s.Status -eq 'Valid' -and $s.SignerCertificate.GetNameInfo("
              "[System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,$false) "
              "-eq 'Microsoft Corporation'){exit 0};exit 1")
    return subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', script],
                          env={**os.environ, 'PUYO_INSTALLER': str(target)},
                          creationflags=subprocess.CREATE_NO_WINDOW).returncode == 0


class ShellExecuteInfo(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('fMask', wintypes.ULONG), ('hwnd', wintypes.HWND),
               ('lpVerb', wintypes.LPCWSTR), ('lpFile', wintypes.LPCWSTR),
               ('lpParameters', wintypes.LPCWSTR), ('lpDirectory', wintypes.LPCWSTR),
               ('nShow', ctypes.c_int), ('hInstApp', wintypes.HINSTANCE),
               ('lpIDList', ctypes.c_void_p), ('lpClass', wintypes.LPCWSTR),
               ('hkeyClass', wintypes.HKEY), ('dwHotKey', wintypes.DWORD),
               ('hIcon', wintypes.HANDLE), ('hProcess', wintypes.HANDLE)]


def install(target: Path, arguments: str) -> int:
    """ShellExecuteでWindowsの昇格確認を利用する。"""
    info = ShellExecuteInfo()
    info.cbSize, info.fMask = ctypes.sizeof(info), SEE_MASK_NOCLOSEPROCESS
    info.lpVerb, info.lpFile, info.lpParameters, info.nShow = 'open', str(target), arguments, SW_SHOWNORMAL
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        raise ctypes.WinError()
    kernel = ctypes.windll.kernel32
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        kernel.WaitForSingleObject(info.hProcess, INFINITE)
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
            raise ctypes.WinError()
        return code.value
    finally:
        kernel.CloseHandle(info.hProcess)


def verified_install(target: Path, arguments: str) -> int:
    """検証から実行終了まで取得物の書き換えと削除を拒否する。"""
    kernel = ctypes.windll.kernel32
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(target), GENERIC_READ, FILE_SHARE_READ, None,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError()
    try:
        if not verify_microsoft(target):
            raise ValueError('Microsoft signature verification failed')
        return install(target, arguments)
    finally:
        kernel.CloseHandle(handle)


def ensure_runtime(name: str, url: str, arguments: str, present: Callable[[], bool]) -> None:
    try:
        if present():
            return
        prompt = text(f'{name} がありません。Microsoft の公式サイトから入れますか？',
                      f'{name} was not found. Install it from Microsoft?')
        if url == WEB_URL:
            prompt += text('\n入れなくてもブラウザで開けます。', '\nWithout it, the app opens in your browser.')
        if not message(prompt, True):
            return
        with tempfile.TemporaryDirectory(prefix='puyoZeus-') as directory:
            target = Path(directory) / 'MicrosoftRuntime.exe'
            download(url, target)
            code = verified_install(target, arguments)
            if code not in SUCCESS_CODES:
                raise OSError(f'Installer exit code: {code}')
            if code:
                message(text('Windows を再起動してください。', 'Please restart Windows.'))
            if not present():
                message(text(f'{name} がまだ利用できません。起動を続けます。',
                             f'{name} is still unavailable. Continuing startup.'))
    except Exception as error:
        message(text(f'{name} を入れられませんでした。起動を続けます。\n',
                     f'{name} could not be installed. Continuing startup.\n') + str(error))


def is_onedrive(root: Path) -> bool:
    full = str(root.absolute()).casefold().rstrip('\\/') + os.sep
    for name in ('OneDrive', 'OneDriveConsumer', 'OneDriveCommercial'):
        value = os.environ.get(name)
        if value and full.startswith(str(Path(value).absolute()).casefold().rstrip('\\/') + os.sep):
            return True
    return any(p.casefold() == 'onedrive' or p.casefold().startswith('onedrive - ') for p in root.parts)


def warn_onedrive(root: Path) -> None:
    marker = root / 'logs/launcher-onedrive-dismissed'
    if not is_onedrive(root) or marker.exists():
        return
    dismiss = message(text('OneDrive の同期で動作が不安定になることがあります。同期しない場所へ展開してください。'
                           '\n起動は続けます。次回からこの注意を表示しませんか？',
                           'OneDrive sync may interfere with the app. Extract outside OneDrive.'
                           '\nStartup will continue. Do not show this warning again?'), True)
    if dismiss:
        try:
            marker.parent.mkdir(exist_ok=True)
            marker.write_text('1', encoding='ascii')
        except OSError:
            pass


def failure(root: Path, reason: str) -> None:
    message(text('起動できませんでした。Visual C++ ランタイムと ZIP 全体の展開を確認してください。\n',
                 'Could not start the app. Check Visual C++ and extract the entire ZIP.\n')
            + reason + '\nLogs: ' + str(root / 'logs'))


def run_python(root: Path, args: list[str], uninstall: bool) -> int:
    executable = root / 'python' / ('python.exe' if uninstall else 'pythonw.exe')
    module = 'distribution_uninstall' if uninstall else 'distribution_app'
    command = [str(executable), '-s', '-m', f'src.phase_j.{module}', '--root', str(root)]
    command += ['--interactive'] if uninstall else (['--autostart'] if '--autostart' in args else [])
    process = subprocess.Popen(command, cwd=root, creationflags=0 if uninstall else subprocess.CREATE_NO_WINDOW)
    try:
        code = process.wait(timeout=None if uninstall else STARTUP_SECONDS)
    except subprocess.TimeoutExpired:
        return 0
    if code and not uninstall:
        failure(root, f'Python exit code: {code}')
    return code


def main(root: Path, args: list[str], uninstall: bool = False) -> int:
    try:
        if not uninstall:
            from update_recovery import recover_before_launch
            recover_before_launch(root)
            from legacy_cleanup import cleanup
            cleanup(root)
            ensure_runtime('Microsoft Visual C++', VC_URL, '/install /passive /norestart', has_vc)
            ensure_runtime('Microsoft Edge WebView2', WEB_URL, '/silent /install', has_webview)
            warn_onedrive(root)
        return run_python(root, args, uninstall)
    except Exception as error:
        if uninstall:
            print(str(error), flush=True)
        else:
            failure(root, str(error))
        return 1
    finally:
        if uninstall:
            try:
                input(text('Enter で閉じます。', 'Press Enter to close.'))
            except EOFError:
                pass
