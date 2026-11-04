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
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_REPORT_BYTES = 20 * 1024
MAX_RECEIPT_BYTES = 8192
MAX_REPORTS = 20
LOG_LINES = 100
MAX_LOG_BYTES = 64 * 1024
SEND_TIMEOUT = 10
PREVIEW_TTL = 15 * 60
RELAY_PLACEHOLDER = 'https://puyolive-reports.example.invalid/api/reports'
SECRET_FIELD = re.compile(r'password|passwd|secret|token|api[_-]?key', re.IGNORECASE)
MIN_IDENTITY_LENGTH = 3


def redact(value: Any, identities: tuple[str, ...] | None = None, root: Path | None = None) -> Any:
    """構造・キーを含めて伏せ字にし、別ユーザーのホームパスも隠す。"""
    identities = identities or (getpass.getuser(), socket.gethostname(),
                                 os.environ.get('COMPUTERNAME', ''))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if SECRET_FIELD.search(str(key)):
                result[str(key)] = '[redacted]'
            else:
                result[redact(str(key), identities, root)] = redact(item, identities, root)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, identities, root) for item in value]
    if not isinstance(value, str):
        return value
    if root is not None:
        pattern = r'[\\/]+'.join(re.escape(part) for part in re.split(r'[\\/]+', str(root).rstrip('\\/')))
        value = re.sub(pattern + r'(?=$|[\\/\s"\'])', '<install>', value, flags=re.IGNORECASE)
    value = re.sub(r'(?i)([\\/]+Users[\\/]+|[\\/]+home[\\/]+)[^\\/\s"\']+', r'\1[redacted]', value)
    value = re.sub(r'(?i)OneDrive - [^\\/"\'\r\n]+', 'OneDrive - [redacted]', value)
    if '@' in value:
        value = re.sub(r'(?i)[\w.%+-]{1,64}@[\w.-]{1,253}\.[a-z]{2,63}', '[redacted]', value)
    value = re.sub(r'(?<![\w:])[\\/]{2,}[^\\/\s"\']+[\\/]+[^\\/\s"\']+', '<network>', value)
    value = re.sub(r'''(?i)\b(password|passwd|secret|token|api[_-]?key)(\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''', r'\1\2[redacted]', value)
    for identity in sorted(set(identities), key=len, reverse=True):
        if len(identity) >= MIN_IDENTITY_LENGTH:
            value = re.sub(r'(?<!\w)' + re.escape(identity) + r'(?!\w)', '[redacted]', value, flags=re.IGNORECASE)
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


def receipt_counts(response: Any) -> dict[str, int]:
    """応答の任意文字列は表示せず、受理件数と重複件数だけを採る。"""
    try:
        data = json.loads(response.read(MAX_RECEIPT_BYTES))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in ('accepted', 'duplicates')
            if type(data.get(key)) is int and 0 <= data[key] <= MAX_REPORTS}


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """数値要約を残し、大きい既存の補助情報から省略する。省略はプレビューにも出す。
    version は受け口の必須欄 (version.app) なので省略しない。"""
    report['omitted_fields'] = []
    for key in ('recent_logs', 'installation_logs', 'self_diagnostics', 'error', 'settings', 'environment', 'summary'):
        if len(json.dumps(report, ensure_ascii=False).encode('utf-8')) <= MAX_REPORT_BYTES:
            break
        report['omitted_fields'].append(key)
        report[key] = [] if key.endswith('logs') else {} if isinstance(report[key], dict) else ''
    return report


class NoRedirect(HTTPRedirectHandler):
    """確認した送信先以外へのリダイレクトを拒否する。"""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


class ReportStore:
    """未送信キューと、一回限り・期限つきの確認用スナップショット。"""

    def __init__(self, directory: Path, fallback_directory: Path | None = None, root: Path | None = None) -> None:
        self.directory = directory
        self.root = root
        self.last_send_result: dict[str, Any] = {}
        self.fallback_directory = fallback_directory
        directory.mkdir(parents=True, exist_ok=True)
        identity = directory / 'installation_id.txt'
        if not identity.exists():
            identity.write_text(secrets.token_hex(16), encoding='ascii')
        self.installation_id = identity.read_text(encoding='ascii').strip()
        self.previews: dict[str, tuple[float, dict[str, Any], list[Path], str]] = {}

    def report(self, version: dict[str, Any], settings: dict[str, Any], error: dict[str, Any],
               diagnostics: dict[str, Any], logs: list[str], install_logs: list[str],
               numeric_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        safe_error = redact(error, root=self.root)
        source = str(safe_error.get('type', 'manual')) + ':' + str(safe_error.get('location', 'settings'))
        report = redact(dict(schema_version=1, report_id=secrets.token_hex(16),
            created_at=time.time(), installation_id=self.installation_id,
            version=version, environment=environment(), settings=settings, error=safe_error,
            fingerprint=hashlib.sha256(source.encode()).hexdigest(),
            summary=source, recent_logs=logs, installation_logs=install_logs,
            self_diagnostics=diagnostics), root=self.root)
        if numeric_summary is not None:
            report['numeric_summary'] = redact(numeric_summary, root=self.root)
            report = compact_report(report)
        return report

    def preview(self, report: dict[str, Any], endpoint: str,
                max_reports: int | None = None) -> dict[str, Any]:
        self.previews = {k: v for k, v in self.previews.items() if time.time() - v[0] < PREVIEW_TTL}
        files = sorted(self.directory.glob('pending-*.json'))[-MAX_REPORTS:]
        if self.fallback_directory and self.fallback_directory.exists():
            files += sorted(self.fallback_directory.glob('pending-*.json'))[-MAX_REPORTS:]
        stored = self.read_pending(files)
        files = [path for path, _ in stored]
        reports = [item for _, item in stored]
        reports = list({r['report_id']: r for r in reports}.values())
        reports.append(report)
        if max_reports is not None:
            if not 1 <= max_reports <= MAX_REPORTS:
                raise ValueError('Invalid report limit')
            reports = reports[-max_reports:]
            selected = {item['report_id'] for item in reports}
            files = [path for path, item in stored if item['report_id'] in selected]
        token = secrets.token_urlsafe(32)
        payload = dict(schema_version=1, reports=reports)
        self.previews[token] = (time.time(), payload, files, endpoint)
        local_only = (urlsplit(endpoint).hostname or '').endswith('.invalid')
        return dict(token=token, destination=endpoint, payload=payload, local_only=local_only)

    def read_pending(self, files: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
        stored = []
        for path in files:
            try:
                item = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(item, dict) or not isinstance(item.get('report_id'), str) or not item['report_id']:
                    raise ValueError('Invalid pending report')
                stored.append((path, redact(item, root=self.root)))
            except (OSError, ValueError):
                try:
                    path.rename(path.with_name('invalid-' + secrets.token_hex(16) + '.json'))
                except OSError:
                    pass
        return stored

    def send(self, token: str, confirmed: bool, allow_local: bool = False) -> bool:
        if confirmed is not True or token not in self.previews:
            raise ValueError('送信内容を確認してください / Review the report first')
        self.last_send_result = {}
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
                self.last_send_result = receipt_counts(response)
        except HTTPError as error:
            self.last_send_result = dict(status=error.code)
            self.save(payload['reports'])
            return False
        except Exception:
            self.save(payload['reports'])
            return False
        for path in files:
            path.unlink(missing_ok=True)
        return True

    def save(self, reports: list[dict[str, Any]]) -> None:
        for report in reports:
            filename = 'pending-' + hashlib.sha256(str(report['report_id']).encode()).hexdigest() + '.json'
            temporary = self.directory / (filename + '.' + secrets.token_hex(8) + '.tmp')
            temporary.write_text(json.dumps(redact(report, root=self.root), ensure_ascii=False), encoding='utf-8')
            temporary.replace(self.directory / filename)
        paths = sorted(self.directory.glob('pending-*.json'), key=lambda p: p.stat().st_mtime)
        for old in paths[:-MAX_REPORTS]:
            old.unlink()
