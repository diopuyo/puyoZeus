"""配布用の写しから、ビルド機のパスだけを取り除く。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

ABSOLUTE_PREFIX = re.compile(r'^(?:file:/+)?(?:[A-Za-z]:[/\\]|/mnt/[a-z]/|/home/|/Users/)', re.I)
USER_ROOT = re.compile(r'^(?:file:/+)?(?:[A-Za-z]:[/\\]Users[/\\][^/\\]+|/mnt/[a-z]/Users/[^/]+|/home/[^/]+)[/\\]?', re.I)
WORK_ROOT = re.compile(r'^(?:file:/+)?(?:[A-Za-z]:[/\\]|/mnt/[a-z]/)puyo1_work[/\\]?', re.I)


def portable_path(value: str) -> str:
    """来歴のパスを匿名のビルドルートからの名前にする。"""
    if not ABSOLUTE_PREFIX.match(value) and not value.startswith('<build>/'):
        return value
    relative = USER_ROOT.sub('', value)
    relative = WORK_ROOT.sub('', relative)
    if relative == value:
        relative = re.sub(r'^(?:file:/+)?(?:[A-Za-z]:[/\\]|/mnt/[a-z]/|/)', '', value)
    relative = relative.replace('\\', '/').removeprefix('<build>/')
    relative = re.sub(r'^.*?(?:\.gemini/antigravity/scratch/|\.codex/worktrees/[^/]+/)?puyo_analyzer/', '', relative)
    return '<build>/' + relative


def portable_provenance(value: Any) -> Any:
    """項目名と数値を保ち、パス値および来歴のパス索引を匿名化する。"""
    if isinstance(value, str):
        return portable_path(value)
    if isinstance(value, list):
        return [portable_provenance(item) for item in value]
    if isinstance(value, dict):
        result = {portable_path(key): portable_provenance(item) for key, item in value.items()}
        if len(result) != len(value):
            raise ValueError('来歴パスの匿名化で索引が重複しました')
        return result
    return value


def sanitize_bundle_provenance(bundle: Path) -> None:
    """モデルの来歴とインストール元URLだけを写しの中で更新する。"""
    paths = list((bundle / 'app/models').rglob('*manifest*.json'))
    paths += list((bundle / 'python/Lib/site-packages').glob('*.dist-info/direct_url.json'))
    for path in paths:
        original = json.loads(path.read_text(encoding='utf-8-sig'))
        sanitized = portable_provenance(original)
        if original != sanitized:
            path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def refresh_model_hashes(bundle: Path, hashes: dict[str, str]) -> dict[str, str]:
    """包装後の中身からRELEASEの資産SHAを作る。"""
    return {name: hashlib.sha256((bundle / 'app/models' / name).read_bytes()).hexdigest()
            for name in hashes}
