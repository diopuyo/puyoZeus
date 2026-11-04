"""解析ごとに出力を退避し、同じ回の排他的な書き込みを保つ。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import uuid

MAX_HISTORY_RUNS = 3
MAX_HISTORY_BYTES = 512 * 1024 * 1024
HISTORY_NAME = re.compile(r'run-\d{8}T\d{12}Z-[0-9a-f]{32}')


def checked_tree(path: Path, root: Path) -> list[Path]:
    """移動・削除の前にリンクと展開先の外への経路を拒否する。"""
    pending, items = [path], []
    while pending:
        item = pending.pop()
        if item.is_symlink() or item.is_junction() or not item.resolve().is_relative_to(root):
            raise OSError('出力先のリンクは使用できません / Output links are not supported')
        items.append(item)
        if item.is_dir():
            pending.extend(item.iterdir())
    return items


def prepare_output(root: Path) -> None:
    """旧版の残りも一括退避し、直近3回・合計512MiB以内を残す。"""
    root = root.resolve()
    output, history = root / 'output', root / 'output_history'
    checked_tree(output, root)
    checked_tree(history, root)
    if output.exists() and any(output.iterdir()):
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.rename(history / f'run-{stamp}-{uuid.uuid4().hex}')
    output.mkdir(exist_ok=True)
    runs = sorted(p for p in history.glob('run-*') if p.is_dir() and HISTORY_NAME.fullmatch(p.name))
    sizes = {p: sum(f.stat().st_size for f in checked_tree(p, root) if f.is_file()) for p in runs}
    while runs and (len(runs) > MAX_HISTORY_RUNS or sum(sizes[p] for p in runs) > MAX_HISTORY_BYTES):
        old = runs.pop(0)
        checked_tree(old, root)
        shutil.rmtree(old)
