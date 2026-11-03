"""送信は利用者の確認後だけ。匿名化済みプレビューをそのまま送る。"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_REPORTS = 20
LOG_LINES = 100
MAX_LOG_BYTES = 64 * 1024
SEND_TIMEOUT = 10
PREVIEW_TTL = 15 * 60
RELAY_PLACEHOLDER = 'https://puyolive-reports.example.invalid/api/reports'


def redact(value: Any, identities: tuple[str, ...] | None = None) -> Any:
    """構造・キーを含めて伏せ字にし、別ユーザーのホームパスも隠す。"""
    identities = identities or (getpass.getuser(), socket.gethostname(),
                                 os.environ.get('COMPUTERNAME', ''))
    if isinstance(value, dict):
        return {redact(str(k), identities): redact(v, identities) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, identities) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r'(?i)([\\/]Users[\\/]|[\\/]home[\\/])[^\\/\s"\']+', r'\1[redacted]', value)
    for identity in sorted(set(identities), key=len, reverse=True):
        if identity:
            value = re.sub(re.escape(identity), '[redacted]', value, flags=re.IGNORECASE)
    return value


def tail(path: Path) -> list[str]:
    """大きなログ全体をメモリへ読み込まない。"""
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - MAX_LOG_BYTES))
        return stream.read(MAX_LOG_BYTES).decode('utf-8', errors='replace').splitlines()[-LOG_LINES:]


def environment() -> dict[str, Any]:
    """端末名を含むplatform文字列は採らず、取得不能項目も明示する。"""
    result: dict[str, Any] = dict(os=platform.system(), os_version=platform.version(),
        cpu=platform.processor(), cpu_count=os.cpu_count(), gpu='unknown', memory_bytes=None)
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page', 'free_page', 'virtual', 'free_virtual', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result['memory_bytes'] = status.total
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Video') as root:
                names = []
                for index in range(winreg.QueryInfoKey(root)[0]):
                    try:
                        with winreg.OpenKey(root, winreg.EnumKey(root, index) + r'\0000') as key:
                            names.append(winreg.QueryValueEx(key, 'DriverDesc')[0])
                    except OSError:
                        pass
                result['gpu'] = sorted(set(names)) or 'unknown'
        except OSError:
            pass
    return result


class NoRedirect(HTTPRedirectHandler):
    """確認した送信先以外へのリダイレクトを拒否する。"""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class ReportStore:
    """未送信キューと、一回限り・期限つきの確認用スナップショット。"""

    def __init__(self, directory: Path, fallback_directory: Path | None = None) -> None:
        self.directory = directory
        self.fallback_directory = fallback_directory
        directory.mkdir(parents=True, exist_ok=True)
        identity = directory / 'installation_id.txt'
        if not identity.exists():
            identity.write_text(secrets.token_hex(16), encoding='ascii')
        self.installation_id = identity.read_text(encoding='ascii').strip()
        self.previews: dict[str, tuple[float, dict[str, Any], list[Path], str]] = {}

    def report(self, version: dict[str, Any], settings: dict[str, Any], error: dict[str, Any],
               diagnostics: dict[str, Any], logs: list[str], install_logs: list[str]) -> dict[str, Any]:
        safe_error = redact(error)
        source = str(safe_error.get('type', 'manual')) + ':' + str(safe_error.get('location', 'settings'))
        return redact(dict(schema_version=1, report_id=secrets.token_hex(16),
            created_at=time.time(), installation_id=self.installation_id,
            version=version, environment=environment(), settings=settings, error=safe_error,
            fingerprint=hashlib.sha256(source.encode()).hexdigest(),
            summary=source, recent_logs=logs, installation_logs=install_logs,
            self_diagnostics=diagnostics))

    def preview(self, report: dict[str, Any], endpoint: str) -> dict[str, Any]:
        self.previews = {k: v for k, v in self.previews.items() if time.time() - v[0] < PREVIEW_TTL}
        files = sorted(self.directory.glob('pending-*.json'))[-MAX_REPORTS:]
        if self.fallback_directory and self.fallback_directory.exists():
            files += sorted(self.fallback_directory.glob('pending-*.json'))[-MAX_REPORTS:]
        reports = [redact(json.loads(p.read_text(encoding='utf-8-sig'))) for p in files]
        reports = list({r['report_id']: r for r in reports}.values())
        reports.append(report)
        token = secrets.token_urlsafe(32)
        payload = dict(schema_version=1, reports=reports)
        self.previews[token] = (time.time(), payload, files, endpoint)
        local_only = (urlsplit(endpoint).hostname or '').endswith('.invalid')
        return dict(token=token, destination=endpoint, payload=payload, local_only=local_only)

    def send(self, token: str, confirmed: bool, allow_local: bool = False) -> bool:
        if confirmed is not True or token not in self.previews:
            raise ValueError('送信内容を確認してください / Review the report first')
        created, payload, files, endpoint = self.previews.pop(token)
        if time.time() - created > PREVIEW_TTL:
            raise ValueError('確認期限切れ / Preview expired')
        parsed = urlsplit(endpoint)
        if (parsed.hostname or '').endswith('.invalid'):
            self.save(payload['reports'])
            return False
        valid = parsed.scheme == 'https' and bool(parsed.hostname) and not parsed.username and not parsed.password
        valid |= allow_local and parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
        try:
            if not valid:
                raise ValueError('HTTPS中継URLが必要です')
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            request = Request(endpoint, body, {'Content-Type': 'application/json'}, method='POST')
            with build_opener(NoRedirect()).open(request, timeout=SEND_TIMEOUT) as response:
                if not 200 <= response.status < 300:
                    raise OSError('中継が受理しませんでした')
        except Exception:
            self.save(payload['reports'])
            return False
        for path in files:
            path.unlink(missing_ok=True)
        return True

    def save(self, reports: list[dict[str, Any]]) -> None:
        for report in reports:
            filename = 'pending-' + hashlib.sha256(str(report['report_id']).encode()).hexdigest() + '.json'
            (self.directory / filename).write_text(json.dumps(redact(report), ensure_ascii=False), encoding='utf-8')
        paths = sorted(self.directory.glob('pending-*.json'), key=lambda p: p.stat().st_mtime)
        for old in paths[:-MAX_REPORTS]:
            old.unlink()
