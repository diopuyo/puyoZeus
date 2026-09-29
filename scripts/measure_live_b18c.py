"""同じ観測をB18bと最適化版へ交互順に渡し、追加窓処理だけを比較する。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np

from scripts.profile_live_b18c import profiled_baseline, VIDEO, PROFILE_FRAMES, PROFILE_START_SEC, OUT
from scripts.verify_live_b18a import quantiles
from src.phase_j.live_snapshot import LiveSnapshotInputs


def paired_overlay() -> type:
    old = profiled_baseline()
    class Paired(LiveSnapshotInputs):
        def __init__(self, reader: Any) -> None:
            super().__init__(reader)
            self.before = old(reader)
            # 計測対象外の死亡画像読取を重ねない。新経路の判定だけを通知へ載せる。
            self.before.terminal = SimpleNamespace(update=lambda frame: ())
            self.paired_rows, self.paired_equal = 0, 0
            self.first_difference: dict | None = None
            self.recent: list = []

        def update(self, frame: np.ndarray, result: Any, stamp: float) -> Any:
            raw_sec = self.reader._live_raw_board_sec
            hsv = self.reader._live_hsv_boards
            results = {}
            order = ('before', 'after') if self.paired_rows % 2 == 0 else ('after', 'before')
            for name in order:
                self.reader._live_raw_board_sec = raw_sec
                self.reader._live_hsv_boards = hsv.copy()
                method = self.before.update if name == 'before' else super().update
                results[name] = method(frame, result, stamp)
            self.reader._live_raw_board_sec, self.reader._live_hsv_boards = raw_sec, hsv
            same = same_inputs(results['before'], results['after'])
            self.paired_rows += 1
            self.paired_equal += same
            if not same and self.first_difference is None:
                self.first_difference = dict(row=self.paired_rows, t_sec=stamp)
                raise AssertionError(f'窓入力がB18bと不一致: {self.first_difference}')
            self.progress(stamp)
            self.cost.update({'before_'+key: value for key, value in self.before.cost.items()})
            self.cost.update({'stage_'+key: value for key, value in self.before.stages.items()})
            return results['after']

        def progress(self, stamp: float) -> None:
            self.recent.append((self.before.cost['snapshot_sec'], self.cost['snapshot_sec'],
                                len(self.reader._live_hsv_pixels) == 2))
            if len(self.recent) == 1000:
                print(json.dumps(dict(t_sec=stamp, rows=self.paired_rows,
                    before=quantiles([r[0] for r in self.recent]), after=quantiles([r[1] for r in self.recent]),
                    reused_pixels=sum(r[2] for r in self.recent))), flush=True)
                self.recent.clear()
    return Paired


def same_inputs(before: Any, after: Any) -> bool:
    for name in ('p1', 'p2'):
        old, new = getattr(before, name), getattr(after, name)
        if old.prefire_snapshot != new.prefire_snapshot:
            return False
        a, b = old.midchain_board, new.midchain_board
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a._grid, b._grid)):
            return False
    return True


def micro() -> None:
    from src.image_reader import ImageReader
    from src.board import Board
    cv2.setNumThreads(1)
    reader = ImageReader()
    reader.enable_native_hsv()
    values = (profiled_baseline()(reader), LiveSnapshotInputs(reader))
    side, rows = SimpleNamespace(cnn_board=Board()), []
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(PROFILE_START_SEC*cap.get(cv2.CAP_PROP_FPS)))
    for index in range(PROFILE_FRAMES):
        ok, frame = cap.read()
        if not ok:
            raise EOFError(index)
        reader._live_hsv_pixels = {idx: cv2.cvtColor(frame[r.y:r.y+r.height, r.x:r.x+r.width], cv2.COLOR_BGR2HSV)
            for idx, r in enumerate((reader._p1_region, reader._p2_region))}
        row = {}
        for version in ((0, 1) if index % 2 == 0 else (1, 0)):
            value = values[version]
            value.cost = dict(hsv_sec=0.)
            started = perf_counter()
            for idx in range(2):
                value.retain(frame, side, idx, index/30)
            row[str(version)] = perf_counter()-started
        rows.append(row)
        cap.grab()
    cap.release()
    report = {key: quantiles([row[key] for row in rows]) for key in rows[0]}
    (OUT/'micro_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    micro()
