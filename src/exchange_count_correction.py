"""読み取り到着後だけに公開する残存数補正。初回予測を書き換えない。"""
from __future__ import annotations

from typing import Any
from src.exchange_count_prediction import histogram, simulate, validate_times, COLS
from src.exchange_count_v34 import grounded

PAIR_SIZE = 2

class RemainingCorrection:
    def __init__(self, input_sec: float) -> None:
        self.input_sec = input_sec
        self.latest = input_sec
        self.result: dict | None = None

    def observe(self, stamp: float, state: str, normal_score: bool,
                sources: list[dict], colors: list[int], native: Any,
                allow_visual_stable: bool = False) -> dict | None:
        if stamp < self.latest:
            raise ValueError('補正通知は時刻順で渡してください')
        validate_times(stamp, sources)
        self.latest = stamp
        if (self.result is not None or stamp <= self.input_sec or not normal_score
                or (state != 'STABLE' and not allow_visual_stable)):
            return None
        # 既存投票分類器の連続2画像が同じ接地盤を示した時点を最初の可読盤とする。
        images = [s for s in sources if s['kind'] == 'image' and s.get('classifier') == 'vote']
        if len(images) < 2 or images[-2]['t'] >= images[-1]['t']:
            return None
        boards = [grounded(s['board']) for s in images[-2:]]
        board = boards[-1]
        # 操作中の1組を超える浮遊物は、まだ落下中か演出中なので可読としない。
        if any(sum(a != b for a, b in zip(s['board'], b)) > PAIR_SIZE for s, b in zip(images[-2:], boards)):
            return None
        if boards[0] != board or any(c not in (0, 9, *colors) for c in board):
            return None
        if any(board[:COLS]) or simulate(board, native)['erased']:
            return None
        self.result = dict(input_sec=stamp, evidence_times=[s['t'] for s in images[-2:]],
            remaining=histogram(board), board=board, kind='連鎖後観測補正',
            note='初回予測と別採点。隠し段は画像だけでは確定しない。')
        return self.result.copy()
