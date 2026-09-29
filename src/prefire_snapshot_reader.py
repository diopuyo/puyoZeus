"""現在層から独立した発火直前画像の窓と多数決。"""
from __future__ import annotations
from collections import deque
from typing import Any
import cv2
import numpy as np
from src.animation_filter import AnimationFilter
from src.board import Board, COLOR_OJAMA, COLOR_UNKNOWN, BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS
from src.effect_glow_detector import is_effect_glow_active
from src.match_color_evidence import GAME_COLOR_COUNT
from src.midchain_board_reader import MidchainBoardReader

WINDOW_SEC = 1.2
MIN_FRAMES = 3
MAX_CORRECTION_CELLS = 4
PAIR_SIZE = 2
FRAME_SIZE = (1920, 1080)


def grounded(grid: np.ndarray) -> bool:
    """各列の最上ぷよより下に空隙がないことを確認する。"""
    return all(not np.any(grid[np.flatnonzero(grid[:, c])[0]:, c] == 0)
               for c in range(grid.shape[1]) if np.any(grid[:, c]))


def vote_window(frames: list[dict], trigger: float) -> dict:
    """最後に着地形状が連続した区間だけを選び、消去・浮遊・品質不良を除く。"""
    runs: list[list[dict]] = []
    interrupted = True
    for frame in frames:
        if not trigger-WINDOW_SEC <= frame['t_sec'] < trigger:
            continue
        grid = np.asarray(frame['cnn'])
        if frame['quality'] or not grounded(grid):
            interrupted = True
            continue
        if interrupted or not np.array_equal(grid != 0, np.asarray(runs[-1][-1]['cnn']) != 0):
            runs.append([])
        runs[-1].append(frame)
        interrupted = False
    valid = [r for r in runs if len(r) >= MIN_FRAMES]
    if not valid:
        return dict(reason='no_landed_window', board=None)
    # 消去後の少数セル盤面を誤採用しない。最大占有数が最後に成立した着地区間。
    run = max(valid, key=lambda r: (np.count_nonzero(r[-1]['cnn']), r[-1]['t_sec']))
    stacks = np.array([f['cnn'] for f in run])
    votes = np.stack([(stacks == color).sum(axis=0) for color in range(COLOR_UNKNOWN+1)])
    grid = votes.argmax(axis=0)
    grid[votes.max(axis=0) * 2 <= len(run)] = COLOR_UNKNOWN
    return dict(reason=None, board=grid.tolist(), start_sec=run[0]['t_sec'],
                end_sec=run[-1]['t_sec'], frames=len(run))


class PrefireSnapshotReader:
    """記録補完とライブ表示が共有する独立した読取器。"""

    def __init__(self, reader: Any = None) -> None:
        self.reader = reader if reader is not None else MidchainBoardReader().reader
        self.filters = (AnimationFilter(), AnimationFilter())
        self.frames: tuple[deque, deque] = (deque(), deque())
        self.game: int | None = None

    def read_frame(self, frame: np.ndarray, idx: int, stamp: float) -> dict:
        """CNN融合とHSV単独を同時に読み、既存アニメ品質判定を保存する。"""
        if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        region = (self.reader._p1_region, self.reader._p2_region)[idx]
        quality = self.filters[idx].is_animation(frame, (region.x, region.y, region.width, region.height))
        cnn = self.reader.read_board(frame, region, skip_tier1=True)
        hsv = self.reader.read_board_hsv_only(frame, region)
        glow = is_effect_glow_active(frame, region, frozenset(range(HIDDEN_ROWS, BOARD_ROWS)))
        return dict(t_sec=stamp, cnn=cnn._grid.tolist(), hsv=hsv._grid.tolist(),
                    quality=quality.reason or ('effect_glow' if glow else ''))

    def update(self, frame: np.ndarray, sides: tuple, stamp: float, game: int) -> tuple:
        """試合ごとに画像履歴を消去し、発火通知時だけ直前の窓を出す。"""
        if self.game != game:
            self.game = game
            for buffer, quality in zip(self.frames, self.filters):
                buffer.clear()
                quality.reset()
        snapshots = []
        for idx, side in enumerate(sides):
            event = side.chain_event
            snapshots.append(vote_window(list(self.frames[idx]), event.trigger_sec) if event else None)
            self.frames[idx].append(self.read_frame(frame, idx, stamp))
            while self.frames[idx] and self.frames[idx][0]['t_sec'] < stamp-WINDOW_SEC:
                self.frames[idx].popleft()
        return tuple(snapshots)


def validate_snapshot(origin: Board | None, snapshot: dict | None, colors: tuple) -> tuple[str | None, Board | None]:
    """一組追加または4セル以内の修正だけを許し、不可視段を補完しない。"""
    if snapshot is None or snapshot.get('board') is None:
        return (snapshot or {}).get('reason', 'missing_snapshot'), None
    grid = np.asarray(snapshot['board'])
    if grid.shape != (BOARD_ROWS, BOARD_COLS):
        return 'invalid_shape', None
    if np.any(grid == COLOR_UNKNOWN):
        return 'unknown', None
    if len(colors) != GAME_COLOR_COUNT or not set(np.unique(grid)) <= {0, COLOR_OJAMA, *colors}:
        return 'palette', None
    if not grounded(grid):
        return 'floating', None
    if origin is None:
        return 'missing_origin', None
    cells = np.argwhere(grid != origin._grid)
    additions = len(cells) == PAIR_SIZE and all(origin._grid[r,c] == 0 and grid[r,c] in colors for r,c in cells)
    adjacent = additions and int(np.abs(cells[0]-cells[1]).sum()) == 1
    if not adjacent and len(cells) > MAX_CORRECTION_CELLS:
        return 'origin_difference', None
    board = Board()
    board._grid = grid.astype(np.int8, copy=True)
    return None, board
