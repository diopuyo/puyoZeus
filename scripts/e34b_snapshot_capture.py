"""E32の既存画像窓取得を、新収集の発火時刻に適用する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import cv2

from scripts.enrich_e31_snapshots import capture_window
from src.prefire_snapshot_reader import PrefireSnapshotReader


class SnapshotCapture:
    """保持印とは独立に、E31/E32と同じ30Hz画像窓を発火通知時に読む。"""

    def __init__(self, video: Path, dest: Path) -> None:
        self.cap = cv2.VideoCapture(str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.fps > 0
        self.dest, self.game, self.start = dest, None, 0.
        self.windows: dict[str, dict] = {}

    def install(self) -> None:
        """スナップショット読取だけを既存固定入力生成方式へ置換する。"""
        def update(reader: Any, frame: Any, sides: tuple, stamp: float, game: int) -> tuple:
            if game != self.game:
                self.game, self.start = game, stamp
            values = []
            for idx, side in enumerate(sides):
                event = side.chain_event
                if event is None:
                    values.append(None)
                    continue
                key = f'{game}:{idx}:{event.trigger_sec}'
                if key not in self.windows:
                    self.windows[key] = capture_window(reader, self.cap, idx,
                        event.trigger_sec, self.fps, self.start)
                values.append({k: v for k, v in self.windows[key].items() if k != 'raw_frames'})
            return tuple(values)
        PrefireSnapshotReader.update = update

    def close(self) -> None:
        """多数決と原観測を同時に保存し、D1残差に同じ根拠を渡す。"""
        self.cap.release()
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.dest, 'wt', encoding='utf-8') as stream:
            json.dump(self.windows, stream, ensure_ascii=False, separators=(',', ':'))
