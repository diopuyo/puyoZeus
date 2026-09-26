"""全表示フレームから、凍結の割合・最長時間・更新頻度を測る。"""
from __future__ import annotations

import numpy as np

SECONDS_PER_MINUTE = 60


def display_freshness(values: np.ndarray, fps: float, allow_missing: bool = False) -> dict:
    """厳密同値を隣接対で測定する。試合境界も全フレーム母集団に含める。"""
    values = np.asarray(values)
    invalid = np.isinf(values).any() if allow_missing else not np.isfinite(values).all()
    if values.ndim != 1 or not len(values) or invalid:
        raise ValueError("鮮度には非空で有限な一次元の全表示列が必要")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fpsは正の有限値が必要")
    same = values[1:] == values[:-1]
    if allow_missing:
        same |= np.isnan(values[1:]) & np.isnan(values[:-1])
    boundaries = np.r_[0, np.flatnonzero(~same) + 1, len(values)]
    updates = int((~same).sum())
    minutes = len(values) / fps / SECONDS_PER_MINUTE
    return dict(frames=len(values), adjacent_pairs=len(same), equal_pairs=int(same.sum()),
                equal_fraction=float(same.mean()) if len(same) else 0.0,
                longest_equal_seconds=float(np.diff(boundaries).max() / fps),
                updates=updates, minutes=minutes, updates_per_minute=updates / minutes)


def evaluation_freshness(display: dict, mode: str, fps: float) -> dict:
    """平滑化前の評価値で測る。未評価NaN同士は未更新として分母に残す。"""
    if mode not in ("on", "off"):
        raise ValueError("鮮度のモードはon/offのみ")
    column = "display_p1" if mode == "on" else "adv_raw_last"
    values = display[column]
    return dict(display_freshness(values, fps, allow_missing=True),
                column=column, missing_frames=int(np.isnan(values).sum()))
