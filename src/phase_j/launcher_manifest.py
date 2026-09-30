"""配布物の完全性確認 (MANIFEST.json の生成と起動時 SHA256 照合)。

配布ではモデル・設定・ソースが 1 バイトでも欠けると、認識や評価が黙って別物になる
(fail-silent)。起動時に全ファイルの存在・サイズ・SHA256 と主要パッケージの版を照合し、
不一致は名前つきで報告して起動を止める。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

MANIFEST_SCHEMA = 1
MANIFEST_FILENAME = 'MANIFEST.json'
HASH_CHUNK_BYTES = 1 << 20
DIST_INFO_SUFFIX = '.dist-info'
MAX_REPORTED = 20  # 一覧表示は先頭のみ (全件はレポートに保持)
_NORMALIZE = re.compile(r'[-_.]+')


class ManifestError(ValueError):
    """MANIFEST.json 自体が読めない・形式不正。"""


@dataclass
class ManifestReport:
    """照合結果。母数 (checked) を必ず持ち、0 件合格を成功と読み替えさせない。"""

    checked: int = 0
    checked_bytes: int = 0
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)
    package_mismatch: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.checked > 0 and not (
            self.missing or self.size_mismatch or self.hash_mismatch or self.package_mismatch)

    def summary(self) -> str:
        """人が読める要約。失敗時は種別ごとに先頭 MAX_REPORTED 件を示す。"""
        lines = [f'照合 {self.checked} ファイル / {self.checked_bytes / 1e6:.1f} MB']
        for title, items in (('欠落', self.missing), ('サイズ不一致', self.size_mismatch),
                             ('SHA256 不一致', self.hash_mismatch),
                             ('パッケージ版不一致', self.package_mismatch)):
            if items:
                shown = items[:MAX_REPORTED]
                more = f' ほか {len(items) - len(shown)} 件' if len(items) > len(shown) else ''
                lines.append(f'{title} {len(items)} 件: ' + ', '.join(shown) + more)
        if self.checked == 0:
            lines.append('照合対象が 0 件です (MANIFEST が空)')
        return '\n'.join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    """PEP 503 相当の正規化 (scikit_learn と scikit-learn を同一視)。"""
    return _NORMALIZE.sub('-', name).lower()


def installed_versions(site_packages: Path) -> dict[str, str]:
    """site-packages の *.dist-info 名から {パッケージ名: 版} を返す (import しない)。"""
    versions: dict[str, str] = {}
    for info in sorted(site_packages.glob('*' + DIST_INFO_SUFFIX)):
        stem = info.name[:-len(DIST_INFO_SUFFIX)]
        name, _, version = stem.rpartition('-')
        if name:
            versions[normalize_name(name)] = version
    return versions


def build_manifest(root: Path, files: Iterable[Path], packages: dict[str, str] | None = None) -> dict:
    """root 基準の相対パス (POSIX 形式) で files の size/SHA256 を記録する。"""
    entries: dict[str, dict] = {}
    for path in sorted(set(files)):
        rel = path.relative_to(root).as_posix()
        entries[rel] = dict(size=path.stat().st_size, sha256=sha256_file(path))
    return dict(schema=MANIFEST_SCHEMA, files=entries, packages=dict(sorted((packages or {}).items())))


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f'MANIFEST を読めません: {path} ({error})') from error
    if not isinstance(data, dict) or data.get('schema') != MANIFEST_SCHEMA or not isinstance(data.get('files'), dict):
        raise ManifestError('MANIFEST の形式が不正です (schema または files が想定外)')
    return data


def verify_files(root: Path, manifest: dict, report: ManifestReport) -> None:
    """全ファイルの存在→サイズ→SHA256 の順に照合 (安い検査で先に落とす)。"""
    for rel, expected in manifest['files'].items():
        path = root / rel
        if not path.is_file():
            report.missing.append(rel)
            continue
        report.checked += 1
        report.checked_bytes += expected['size']
        if path.stat().st_size != expected['size']:
            report.size_mismatch.append(rel)
        elif sha256_file(path) != expected['sha256']:
            report.hash_mismatch.append(rel)


def verify_packages(manifest: dict, actual: dict[str, str], report: ManifestReport) -> None:
    for name, version in manifest.get('packages', {}).items():
        found = actual.get(normalize_name(name))
        if found != version:
            report.package_mismatch.append(f'{name} 期待={version} 実際={found}')


def verify_manifest(root: Path, manifest_path: Path, site_packages: Path | None = None) -> ManifestReport:
    """root 配下を manifest_path と照合する。site_packages を渡すと版も照合する。"""
    manifest = load_manifest(manifest_path)
    report = ManifestReport()
    verify_files(root, manifest, report)
    if site_packages is not None:
        verify_packages(manifest, installed_versions(site_packages), report)
    return report
