"""B3 記録 (observations / lean) を v4.1x の配信入力形式へ時刻順に変換する (dl1k 研究専用)。

NEXT は B3 と同じく lean の確定行 (自側の STABLE 確定時の NEXT/NEXT2) だけから取る。
lean の勝敗列 `won` は教師なので読まない (読む列を LEAN_FIELDS に固定する)。
各コマでは、そのコマ以前 (frame_idx <= 現在コマ) の lean 行だけを使う。
events.jsonl は配信入力に対応する欄が無いため変換に使わない (出自 SHA だけ記録する)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LEAN_FIELDS = ('frame_idx', 'side', 't_sec', 'game_idx', 'next1_a', 'next1_b', 'dnext_a', 'dnext_b')
FORBIDDEN_LEAN_FIELDS = ('won',)
SIDE_NAMES = ('1P', '2P')
QUEUE_FIELDS = ('next1_a', 'next1_b', 'dnext_a', 'dnext_b')
FIRST_COLOR, LAST_COLOR = 1, 5
CHAIN_FIELDS = ('trigger_sec', 'chain_count', 'total_score', 'mechanism')


@dataclass
class LeanQueues:
    """lean の NEXT 確定行を時刻順に進める外部状態。"""
    frames: np.ndarray
    sides: np.ndarray
    queues: np.ndarray
    cursor: int = 0
    current: list = field(default_factory=lambda: [None, None])

    @classmethod
    def load(cls, path: Path) -> LeanQueues:
        with np.load(path, allow_pickle=False) as data:
            table = {name: np.asarray(data[name]) for name in LEAN_FIELDS}
        order = np.argsort(table['frame_idx'], kind='stable')
        sides = np.array([SIDE_NAMES.index(str(s)) for s in table['side'][order]], dtype=np.int8)
        queues = np.stack([table[k][order] for k in QUEUE_FIELDS], axis=1).astype(int)
        return cls(table['frame_idx'][order].astype(np.int64), sides, queues)

    def advance(self, frame: int) -> None:
        """frame 以前の lean 行を取り込む。未来の行は読まない。"""
        while self.cursor < len(self.frames) and self.frames[self.cursor] <= frame:
            row = self.queues[self.cursor]
            self.current[int(self.sides[self.cursor])] = tuple(
                int(v) if FIRST_COLOR <= int(v) <= LAST_COLOR else 0 for v in row)
            self.cursor += 1

    def queue(self, side: int) -> tuple[int, int, int, int] | None:
        return self.current[side]


@dataclass(frozen=True)
class Frame:
    """配信入力 1 件。index は入力列での位置 (worker の再生先)。"""
    index: int
    frame: int
    t: float
    game: int
    raw: list


def board_value(board: Any) -> dict | None:
    return None if board is None else dict(board=dict(array=board))


def chain_value(chain: dict | None) -> dict | None:
    """B3 の連鎖欄を配信入力の chain_event 名前空間へ写す (値は変えない)。"""
    if chain is None:
        return None
    namespace = {k: chain[k] for k in CHAIN_FIELDS}
    namespace['before_board'] = board_value(chain['before_board'])
    return dict(namespace=namespace)


def side_raw(side: dict, queue: tuple[int, int, int, int] | None) -> dict:
    """1 側の観測を配信入力の形へ。NEXT は lean 由来 (無ければ欠測)。"""
    raw = dict(state=dict(state=side['state']), confirmed_board=board_value(side['confirmed_board']),
               score=side['score'], next_slide_motion=bool(side['next_slide_motion']),
               next_pair=None if queue is None else dict(tuple=queue[:2]),
               dnext_pair=None if queue is None else dict(tuple=queue[2:]))
    chain = chain_value(side.get('chain'))
    if chain is not None:
        raw['chain_event'] = chain
    if os.environ.get('PUYO_EX_RETR') == '1':   # retR の撃った盤 (src/fire_board.py) 用。EX の計算は読まない
        raw['raw_cnn_board'], raw['formula'] = side.get('raw_cnn_board'), side.get('formula')
    if os.environ.get('PUYO_CHAIN_SPLIT_PLACE') == '1':   # 依頼 §4: 置き (tsumo_count の増え) で連鎖を分ける旗の時だけ渡す
        raw['tsumo_count'] = side.get('tsumo_count')
    return raw


def observation_frames(record: Path) -> Iterator[dict]:
    """observations の frame 行をファイル順に返す。"""
    with gzip.open(record/'observations.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item.get('kind') == 'frame':
                yield item


def notifications(record: Path, limit_frame: int | None = None) -> Iterator[Frame]:
    """時刻順の配信入力。時刻が増えない行は記録の異常として止める。"""
    lean, last = LeanQueues.load(record/'lean.npz'), float('-inf')
    for index, item in enumerate(observation_frames(record)):
        if limit_frame is not None and item['frame'] > limit_frame:
            return
        t = float(item['t_sec'])
        if t <= last:
            raise ValueError(f'時刻が増えない観測: frame={item["frame"]}')
        last = t
        lean.advance(int(item['frame']))
        raw = [side_raw(s, lean.queue(i)) for i, s in enumerate(item['sides'])]
        yield Frame(index, int(item['frame']), t, int(item['game_idx']), raw)
