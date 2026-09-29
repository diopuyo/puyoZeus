"""評価子の回収と試合journalの切替を、親のI/O失敗から保護する。"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .live_spool import DiskRows


@contextmanager
def rollback_rows(*rows: DiskRows) -> Iterator[None]:
    """複数spoolの途中失敗では、直前の確定位置へ全て戻す。"""
    saved = [(row, row.stream.tell(), row.count, row.pending) for row in rows]
    try:
        yield
    except Exception:
        for row, position, count, pending in saved:
            row.stream.seek(position)
            row.stream.truncate()
            row.count, row.pending = count, pending
        raise


def commit_history(owner: Any, reply: dict, commands: list[bytes]) -> None:
    """境界の交換回収が成功するまで、前試合を再生可能なまま保持する。"""
    checkpoint = reply.get('checkpoint')
    replacement = None
    try:
        with rollback_rows(owner.archive, owner.diagnostics):
            owner.archive.extend(reply['sealed'])
            owner.diagnostics.extend(reply['diagnostics'])
            if checkpoint is not None:
                path = owner.directory/f'evaluation-journal-{owner.request_id}.pickle'
                replacement = DiskRows(path)
                replacement.extend(commands[checkpoint['index']:])
                replacement.stream.flush()
    except Exception:
        owner.dirty = True
        if replacement is not None:
            replacement.close()
        raise
    if replacement is not None:
        previous, owner.journal = owner.journal, replacement
        owner.boundary_smoothing = checkpoint['smoothing']
        owner.boundary_next_id = checkpoint['next_id']
        try:
            previous.close()
            previous.path.unlink()
        except Exception:
            # 回収自体は確定済み。次回は新試合journalから再構築する。
            owner.dirty = True
            raise


def failure_kind(error: Exception) -> str:
    """発生経路を記録し、評価子以外の障害を連続失敗数へ混入させない。"""
    frames = []
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name == 'log_event':
            return 'logging_error'
        frames.append(Path(trace.tb_frame.f_code.co_filename).stem)
        trace = trace.tb_next
    if any(name in ('live_spool', 'live_retention', 'live_eval_recovery') for name in frames):
        return 'spool_error'
    if any(name in ('live_publish', 'server', 'live_server') for name in frames):
        return 'publication_error'
    return 'processing_error'
