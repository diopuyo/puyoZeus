"""記録の当該フレームだけをliveと同じ入力口へ渡す。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from src.b3_realtime import QueueObservation
from src.b3_source import iter_frames, load_events, load_lean


def replay_inputs(record: Path, max_frame: int | None = None) -> Iterator[tuple]:
    """教師と試合範囲は入力しない。ファイル順に依存せずframeで安定整列する。"""
    lean, events = load_lean(record, max_frame), load_events(record, max_frame)
    indices = np.argsort(lean.frame, kind='stable')
    li = ei = 0
    for frame in iter_frames(record, max_frame):
        queues, found = [], []
        while li < len(indices) and lean.frame[indices[li]] <= frame['frame']:
            i = int(indices[li])
            queues.append(QueueObservation(i, str(lean.side[i]), int(lean.frame[i]),
                float(lean.t_sec[i]), tuple(int(v) for v in lean.queue[i]), int(lean.game[i])))
            li += 1
        while ei < len(events) and events[ei]['frame'] <= frame['frame']:
            found.append(events[ei])
            ei += 1
        yield frame, tuple(found), tuple(queues)
