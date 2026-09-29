"""B18bの保持処理を段別に計測する。同じ原票で前後を比較するため旧版を固定する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import Any
import cv2
import numpy as np

from scripts.verify_live_b18a import quantiles

BASELINE_COMMIT = '9bf9837'
PROFILE_FRAMES = 600
PROFILE_START_SEC = 2580.6
VIDEO = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4')
OUT = Path('logs/live_b18c')


def baseline() -> ModuleType:
    """比較元の実装をgitから読み、作業中のソースを巻き戻さず実行する。"""
    directory = Path('.git').read_text().strip().removeprefix('gitdir: ')
    if os.name != 'nt' and directory[1:3] == ':/':
        directory = '/mnt/'+directory[0].lower()+directory[2:]
    code = subprocess.check_output(['git', '--git-dir', directory, 'show',
                                    f'{BASELINE_COMMIT}:src/phase_j/live_snapshot.py'])
    module = ModuleType('src.phase_j._b18c_baseline')
    exec(compile(code, '<B18b>', 'exec'), module.__dict__)
    return module


def profiled_baseline() -> type:
    old = baseline()
    class Profiled(old.LiveSnapshotInputs):
        def mark(self, key: str, started: float) -> float:
            now = perf_counter()
            self.stages[key] = self.stages.get(key, 0.)+now-started
            return now

        def retain(self, frame: np.ndarray, side: Any, idx: int, stamp: float) -> None:
            if idx == 0:
                self.stages = {}
            started = perf_counter()
            region = (self.reader._p1_region, self.reader._p2_region)[idx]
            crop = frame[region.y:region.y+region.height, region.x:region.x+region.width].copy()
            started = self.mark('crop_copy', started)
            quality = self.filters[idx].is_animation(crop, (0, 0, region.width, region.height)).reason
            started = self.mark('animation', started)
            glow = old.is_effect_glow_active(frame, region, frozenset(range(old.HIDDEN_ROWS, old.BOARD_ROWS)))
            started = self.mark('glow', started)
            board = self.observed_board(side, idx)
            if board is None:
                return
            hsv, hsv_started = None, perf_counter()
            eligible = not quality and not glow and old.grounded(board._grid)
            started = self.mark('eligibility', started)
            if eligible:
                observed = getattr(self.reader, '_live_hsv_boards', {}).get(idx)
                if observed is None:
                    observed = self.reader.read_board_hsv_only(frame, region)
                hsv = observed._grid.tolist()
            self.cost['hsv_sec'] += perf_counter()-hsv_started
            started = self.mark('hsv_read', started)
            self.buffers[idx].append(dict(t_sec=stamp, cnn=board._grid.tolist(), hsv=hsv,
                quality=quality or ('effect_glow' if glow else ''), glow='effect_glow' if glow else ''))
            self.mark('pack_append', started)
    return Profiled


def main() -> None:
    from src.image_reader import ImageReader
    from src.board import Board
    cv2.setNumThreads(1)
    reader = ImageReader()
    reader.enable_native_hsv()
    value = profiled_baseline()(reader)
    side = SimpleNamespace(cnn_board=Board())
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(PROFILE_START_SEC*fps))
    rows = []
    for index in range(PROFILE_FRAMES):
        ok, frame = cap.read()
        if not ok:
            raise EOFError(index)
        value.cost = dict(hsv_sec=0.)
        for idx in range(2):
            value.retain(frame, side, idx, index/30)
        rows.append(value.stages)
        cap.grab()
    cap.release()
    report = {key: quantiles([row[key] for row in rows]) for key in rows[0]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'initial_profile.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
