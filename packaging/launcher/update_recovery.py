"""Python本体を変更しない更新で、製品モジュール読込前に復旧する入口。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def recover_before_launch(root: Path) -> None:
    marker = root / 'logs/update_pending.json'
    if not marker.exists():
        return
    pending = json.loads(marker.read_text(encoding='utf8'))
    if os.environ.get('PUYO_UPDATE_TOKEN') == pending['token']:
        return
    stage = Path(pending['stage']).resolve()
    if stage.parent != root.resolve().parent:
        raise ValueError('Invalid recovery path')
    journal_path = stage / 'transaction.json'
    journal = json.loads(journal_path.read_text(encoding='utf8'))
    payload = Path(journal['payload']).resolve()
    if Path(journal['root']).resolve() != root.resolve() or not payload.is_relative_to(stage):
        raise ValueError('Invalid recovery journal')
    if journal['state'] == 'complete':
        marker.unlink()
        return
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    subprocess.run([str(payload / 'python/python.exe'), str(stage / 'worker.py'),
                    str(journal_path), '0', '--recover'], check=True, cwd=stage,
                   creationflags=flags)
    if os.name == 'nt':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, '更新を完了できなかったため元の版に戻しました。\n'
                                        'The update was rolled back to the previous version.', 'puyoZeus Live', 0)
