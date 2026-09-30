"""修正後の独立一致観測を記録する。監査結果は認識器へ戻さない。"""
from __future__ import annotations
from functools import wraps
import gzip
import json
from pathlib import Path
from typing import Any
from src.placement_signal_runtime import PlacementSignalRuntime


class CorrectionAudit:
    """次の合図までの最初の良質なCNN=HSV一致で、修正セルを照合する。"""

    def __init__(self, path: Path) -> None:
        self.path, self.pending = path, []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
        self.frames = gzip.open(path.with_suffix('.observations.jsonl.gz'), 'wt')
        self.last_frame = -1

    def write(self, row: dict) -> None:
        """欠測も母数へ残し、修正セル一つにつき一行を保存する。"""
        with self.path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def signal(self, row: dict) -> None:
        """後の置き・連鎖と混同しないよう、合図境界で未確認分を打ち切る。"""
        keep = []
        for cell in self.pending:
            if cell['side'] == row.get('side') and cell['frame'] < row['frame']:
                self.write(dict(cell, status='unverified', end_reason='next_signal'))
            else:
                keep.append(cell)
        self.pending = keep
        for cell in row.get('corrections', []):
            self.pending.append(dict(cell, frame=row['frame'], t_sec=row['t_sec'],
                                     side=row['side'], signal=row['signal']))

    def observe(self, runtime: PlacementSignalRuntime, index: int) -> None:
        """修正フレーム自身を使わず、後続の単独一致だけを正誤へ数える。"""
        self.save_frame(runtime, index)
        keep = []
        for cell in self.pending:
            history = runtime.history[cell['side']]
            obs = history[-1] if history else None
            r, c = cell['row'], cell['col']
            if obs is not None and obs.frame == index >= cell['frame'] and obs.erasing:
                self.write(dict(cell, status='unverified', end_reason='erasure'))
                continue
            usable = obs is not None and obs.frame == index > cell['frame']
            usable = usable and not (obs.quality or obs.erasing) and obs.agreement()[r, c]
            if not usable:
                keep.append(cell)
                continue
            value = int(obs.cnn[r, c])
            self.write(dict(cell, status='verified' if value == cell['after'] else 'wrong',
                            observed_frame=index, observed_value=value))
        self.pending = keep

    def save_frame(self, runtime: PlacementSignalRuntime, index: int) -> None:
        """同じ原フレームを二度数えず、固定母数で採点する独立画像原票を残す。"""
        if index == self.last_frame or not all(runtime.history):
            return
        sides = [history[-1] for history in runtime.history]
        if any(obs.frame != index for obs in sides):
            return
        row = dict(frame=index, t=sides[0].stamp, sides=[dict(cnn=o.cnn.tolist(),
                   hsv=o.hsv.tolist(), quality=o.quality, erasing=o.erasing) for o in sides])
        self.frames.write(json.dumps(row, separators=(',', ':'))+'\n')
        self.last_frame = index

    def install(self) -> None:
        """元の画像読取りと修正をそのまま呼び、監査だけを後置する。"""
        record, observe = PlacementSignalRuntime.record, PlacementSignalRuntime.observe
        reset = PlacementSignalRuntime.reset
        @wraps(record)
        def recorded(runtime: Any, row: dict) -> None:
            record(runtime, row)
            self.signal(row)
        @wraps(observe)
        def observed(runtime: Any, pipe: Any, index: int, stamp: float, frame: Any) -> None:
            observe(runtime, pipe, index, stamp, frame)
            self.observe(runtime, index)
        @wraps(reset)
        def reset_observations(runtime: Any) -> None:
            self.end_pending('recognition_reset')
            reset(runtime)
        PlacementSignalRuntime.record, PlacementSignalRuntime.observe = recorded, observed
        PlacementSignalRuntime.reset = reset_observations

    def end_pending(self, reason: str) -> None:
        """試合境界や読取り故障を越えた観測で、前の盤面を採点しない。"""
        for cell in self.pending:
            self.write(dict(cell, status='unverified', end_reason=reason))
        self.pending.clear()

    def close(self) -> None:
        """記録末尾の未確認も除外しない。"""
        self.end_pending('record_end')
        self.frames.close()
