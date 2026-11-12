"""明示操作による更新取得と、展開前の完全性・パス検査。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from .distribution_logs import SITE_USER_AGENT
from .update_transaction import filesystem_path

RELEASE_BASE = 'https://github.com/diopuyo/puyoZeus/releases/download'
DOWNLOAD_HOSTS = {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
CHUNK_BYTES = 256 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACT_BYTES = 8 * 1024**3
MAX_FILES = 50000
TIMEOUT_SECONDS = 10
SHA_PATTERN = re.compile(r'[a-fA-F0-9]{64}')


class Cancelled(Exception):
    """利用者が中断した。"""


def validate_url(url: str, local_origin: str | None = None) -> str:
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError('Invalid update URL')
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if local_origin and origin == local_origin and parsed.hostname == '127.0.0.1':
        return url
    if parsed.scheme != 'https' or parsed.hostname not in DOWNLOAD_HOSTS or parsed.port not in (None, 443):
        raise ValueError('更新元が GitHub Releases ではありません / Invalid update source')
    if parsed.hostname == 'github.com' and not parsed.path.startswith('/diopuyo/puyoZeus/releases/download/'):
        raise ValueError('Invalid release repository')
    return url


class DownloadRedirect(HTTPRedirectHandler):
    def __init__(self, local_origin: str | None = None) -> None:
        self.local_origin = local_origin

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Request | None:
        validate_url(newurl, self.local_origin)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_download(url: str, local_origin: str | None = None) -> Any:
    validate_url(url, local_origin)
    request = Request(url, headers={'User-Agent': SITE_USER_AGENT})
    return build_opener(DownloadRedirect(local_origin)).open(request, timeout=TIMEOUT_SECONDS)


def release_info(data: Any, current: str) -> dict[str, Any]:
    """旧形式のversionだけの応答と、公開済み一覧の両方を読む。"""
    from packaging.version import Version
    entries = data.get('releases', [data]) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError('Invalid release list')
    releases = [r for r in entries if isinstance(r, dict) and r.get('status', 'published') == 'published']
    releases = [r for r in releases if isinstance(r.get('version'), str) and r['version']]
    if not releases:
        return dict(available=False, status='unavailable')
    release = max(releases, key=lambda r: Version(r['version']))
    version = release['version']
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?', version):
        raise ValueError('Invalid release version')
    name = f'puyoZeus-Live-{version}.zip'
    assets = release.get('files', release.get('assets', []))
    asset = next((a for a in assets if isinstance(a, dict) and a.get('name') == name), {})
    url = asset.get('url', asset.get('download_url', release.get('download_url', f'{RELEASE_BASE}/v{version}/{name}')))
    summary = release.get('summary', release.get('changes', release.get('notes', '')))
    if isinstance(summary, dict):
        summary = summary.get('ja', summary.get('en', ''))
    if isinstance(summary, list):
        summary = '\n'.join(str(item) for item in summary)
    return dict(available=Version(version) > Version(current), status='checked', version=version,
                date=str(release.get('date', '')), summary=str(summary)[:MAX_METADATA_BYTES],
                url=str(url), name=name, sha256=asset.get('sha256', release.get('sha256')),
                sums_url=release.get('sha256sums_url', f'{RELEASE_BASE}/v{version}/SHA256SUMS'))


def expected_sha(info: dict, local_origin: str | None = None) -> str:
    value = info.get('sha256')
    if value is None:
        with open_download(info['sums_url'], local_origin) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError('SHA256SUMS too large')
        matches = [line.split() for line in raw.decode('utf8').splitlines()]
        matches = [parts[0] for parts in matches if len(parts) == 2 and parts[1].lstrip('*') == info['name']]
        if len(matches) != 1:
            raise ValueError('ZIP の SHA256 がありません / Missing ZIP checksum')
        value = matches[0]
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError('Invalid SHA256')
    return value.lower()


def download(info: dict, destination: Path, cancel: threading.Event,
             progress: Callable[[int, int], None], local_origin: str | None = None) -> None:
    expected = expected_sha(info, local_origin)
    digest, received = hashlib.sha256(), 0
    try:
        with open_download(info['url'], local_origin) as response, destination.open('xb') as stream:
            total = int(response.headers.get('Content-Length', '0'))
            if total < 0 or total > MAX_ARCHIVE_BYTES:
                raise ValueError('ZIP too large')
            while True:
                if cancel.is_set():
                    raise Cancelled()
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_ARCHIVE_BYTES:
                    raise ValueError('ZIP too large')
                stream.write(chunk)
                digest.update(chunk)
                progress(received, total)
        if cancel.is_set():
            raise Cancelled()
        if total and received != total:
            raise ValueError('ダウンロードが不完全です / Incomplete download')
        if digest.hexdigest() != expected:
            raise ValueError('SHA256 が一致しません。変更していません / SHA256 mismatch; unchanged')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def archive_paths(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, Path]]:
    entries, seen, size = [], set(), 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        parts = path.parts
        invalid = any(p in ('..', '.') or ':' in p or p.endswith((' ', '.')) for p in parts)
        reserved = any(p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))} for p in parts)
        if not parts or path.is_absolute() or '\\' in info.orig_filename or invalid or reserved:
            raise ValueError('Unsafe ZIP path')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('ZIP link is not allowed')
        key = str(path).casefold()
        if key in seen:
            raise ValueError('Duplicate ZIP path')
        seen.add(key)
        size += info.file_size
        if size > MAX_EXTRACT_BYTES or len(seen) > MAX_FILES:
            raise ValueError('ZIP expands too much')
        entries.append((info, Path(*parts)))
    return entries


def extract(archive_path: Path, stage: Path, version: str, cancel: threading.Event) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive_paths(archive)
        needed = sum(info.file_size for info, _ in entries)
        if shutil.disk_usage(stage).free < needed:
            raise OSError('展開用の空き容量がありません / Not enough disk space')
        target = stage / 'payload'
        target.mkdir()
        for info, path in entries:
            if cancel.is_set():
                raise Cancelled()
            destination = filesystem_path(target / path)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
    roots = [target, *[p for p in target.iterdir() if p.is_dir()]]
    roots = [p for p in roots if (p / 'app/RELEASE.json').is_file()]
    if len(roots) != 1:
        raise ValueError('Invalid application ZIP')
    root = roots[0]
    metadata = json.loads((root / 'app/RELEASE.json').read_text(encoding='utf8'))
    if metadata.get('app') != version or not (root / 'puyoZeus Live.exe').is_file():
        raise ValueError('ZIP の版が一致しません / ZIP version mismatch')
    return root
