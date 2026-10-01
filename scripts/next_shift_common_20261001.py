"""148動画の学習原票 (boards_lean) を側・試合ごとの時刻順列へ分ける共通部 (2026-10-01)。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

LEAN_ROOT = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/indicators_v2/"
                 "boards_lean_phase_l_2026-08-11")
QUEUE_KEYS = ("next1_a", "next1_b", "dnext_a", "dnext_b")
SIDES = ("1P", "2P")


def read_raw(path: str | Path) -> dict[str, np.ndarray]:
    """原票を読み、ファイルを閉じてから返す。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def raw_queues(raw: dict[str, np.ndarray]) -> np.ndarray:
    """(行数, 4) の queue 行列 (原票の列順)。"""
    return np.stack([raw[key] for key in QUEUE_KEYS], axis=1).astype(np.int8)


def side_game_indices(raw: dict[str, np.ndarray]) -> Iterator[np.ndarray]:
    """試合・側ごとの行番号を時刻順 (同時刻は原票順) で返す。"""
    for game in np.unique(raw["game_idx"]):
        for side in SIDES:
            ids = np.flatnonzero((raw["game_idx"] == game) & (raw["side"] == side))
            if len(ids):
                yield ids[np.argsort(raw["t_sec"][ids], kind="stable")]


def read_side_sequences(path: str | Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(盤面列, queue列, 原票の行番号) を試合・側ごとに返す。"""
    raw = read_raw(path)
    queues = raw_queues(raw)
    for ids in side_game_indices(raw):
        yield raw["grids"][ids], queues[ids], ids
