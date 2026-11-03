"""平滑化前の評価が短時間で元へ戻る事象を、試合内だけで数える。"""
from __future__ import annotations

import numpy as np

SPIKE_DELTA = 0.15
RETURN_TOLERANCE = 0.05
RETURN_SECONDS = 0.5
SECONDS_PER_MINUTE = 60.0
TIME_EPSILON = 1e-9


def value_spikes(times: np.ndarray, probabilities: np.ndarray,
                 games: np.ndarray) -> dict:
    """隣接差の各起点を数える。復帰フレーム自体は同じ事象へ含める。"""
    events, duration = [], 0.0
    consumed = -1
    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        if games[i] != games[i - 1] or dt <= 0:
            consumed = -1
            continue
        duration += dt
        before, after = probabilities[i - 1:i + 1]
        if i <= consumed or not np.isfinite([before, after]).all():
            continue
        if abs(after - before) + TIME_EPSILON < SPIKE_DELTA:
            continue
        for j in range(i + 1, len(times)):
            if games[j] != games[i] or times[j] - times[i] > RETURN_SECONDS + TIME_EPSILON:
                break
            if abs(probabilities[j] - before) <= RETURN_TOLERANCE + TIME_EPSILON:
                events.append(dict(t_sec=float(times[i]), return_sec=float(times[j]),
                                   before=float(before), peak=float(after), game_idx=int(games[i])))
                consumed = j
                break
    return dict(count=len(events), duration_seconds=duration,
                per_minute=len(events) * SECONDS_PER_MINUTE / duration if duration else 0.0,
                events=events)
