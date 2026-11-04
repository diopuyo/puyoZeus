"""配布済みの名前とSHA256が一致する旧入口だけを片付ける。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


def plain(path: Path) -> bool:
    """ジャンクションを含むすべての再解析ポイントを拒否する。"""
    return not (path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def checked(root: Path, relative: str) -> Path | None:
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        return None
    try:
        if not all(plain(part) for part in (path, *path.parents)):
            return None
        return path if path.is_file() else None
    except OSError:
        return None


def cleanup(root: Path) -> None:
    """利用者の変更は残し、削除を記録できない場合も残す。"""
    hashes = json.loads(Path(__file__).with_name('legacy_hashes.json').read_text('utf-8'))
    logdir = root / 'logs'
    try:
        if not all(plain(part) for part in (root, *root.parents)):
            return
        if logdir.exists() and not plain(logdir):
            return
        logdir.mkdir(exist_ok=True)
        log = logdir / 'launcher-cleanup.jsonl'
        if log.exists() and not plain(log):
            return
        with log.open('a', encoding='utf-8') as stream:
            for relative, allowed in hashes.items():
                path = checked(root, relative)
                if path is None:
                    continue
                try:
                    with path.open('rb') as source:
                        digest = hashlib.file_digest(source, 'sha256').hexdigest()
                    if digest in allowed and checked(root, relative) is not None:
                        path.unlink()
                        stream.write(json.dumps({'removed': relative, 'sha256': digest}, ensure_ascii=False) + '\n')
                        stream.flush()
                except OSError:
                    continue
    except OSError:
        return
