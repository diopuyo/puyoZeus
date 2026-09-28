"""捨てフレームの影響を比較するための、画像を持たない認識監査記録。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BOARD_SHAPE = (2, 13, 6)
MILLISECONDS = 1000.0


class RecognitionAudit:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        from .live_spool import DiskRows, bounded_enabled
        if path is not None and bounded_enabled():
            self.rows = DiskRows(path.parent/'spool'/'recognition.pickle')

    def append(self, notice: Any, frame: Any, wall: float, cpu: float, ready: bool) -> None:
        if self.path is None:
            return
        result = notice.result()
        boards, raw = (np.full(BOARD_SHAPE, -1, dtype=np.int8) for _ in range(2))
        sides = (result.p1, result.p2)
        for i, side in enumerate(sides):
            if side.confirmed_board is not None:
                boards[i] = side.confirmed_board._grid
            if side.cnn_board is not None:
                raw[i] = side.cnn_board._grid
        self.rows.append(dict(frame=notice.frame, t_sec=notice.t_sec, boards=boards, raw=raw,
            stable=[s.state.name == 'STABLE' and s.confirmed_board is not None for s in sides],
            states=[s.state.name for s in sides], active=result.is_match_active, ready=ready,
            scores=[-1 if s.score is None else s.score for s in sides],
            recognition_ms=wall*MILLISECONDS, cpu_ms=cpu*MILLISECONDS, queue_put_ms=0.,
            acquired_at=frame.acquired_at, captured_at=frame.captured_at,
            recognized_at=notice.recognized_at, dropped_before=notice.dropped_before))

    def queue_wait(self, seconds: float) -> None:
        if self.path is not None and self.rows:
            self.rows[-1]['queue_put_ms'] = seconds*MILLISECONDS

    def save(self, runtime: dict, source: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.array([r[key] for r in self.rows]) for key in self.rows[0]} if self.rows else {}
        np.savez_compressed(self.path, **arrays)
        metadata = dict(runtime=runtime, notifications=len(self.rows), dropped=source.dropped,
                        dropped_times=list(getattr(source, 'dropped_times', [])))
        self.path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        if hasattr(self.rows, 'close'):
            self.rows.close()
