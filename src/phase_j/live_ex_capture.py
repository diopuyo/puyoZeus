"""画像を再認識せず、正式OBS版と同じSTABLE時のNEXTをEXへ渡す。"""
from __future__ import annotations

from copy import deepcopy
import os
from typing import Any


ENV_TAPE_NEXT = 'PUYO_LIVE_24H_TAPE_NEXT'


def tape_next_enabled() -> bool:
    """ツモのテープにだけ毎コマの NEXT を送るか (既定 OFF)。"""
    return os.environ.get(ENV_TAPE_NEXT) == '1'


def add_fire_inputs(raw: dict, row: dict, index: int) -> None:
    """retRのときだけ、同じコマのB3観測済みの盤と式を配送する。"""
    if os.environ.get('PUYO_LIVE_24H_RETR') != '1':
        return
    sides = row.get('sides', ())
    if index >= len(sides):
        return
    observed = sides[index]
    from src.chain_continuity import end_enabled, recheck_enabled
    if end_enabled() or recheck_enabled():
        raw.update(chain=deepcopy(observed.get('chain')),
                   continuity_next=observed.get('next_pair'), continuity_dnext=observed.get('dnext_pair'),
                   tsumo_count=observed.get('tsumo_count'), landing_diag=observed.get('landing_diag'))
    if os.environ.get('PUYO_CHAIN_SPLIT_PLACE') == '1':
        raw['tsumo_count'] = observed.get('tsumo_count')
    formula = observed.get('formula') or {}
    raw['raw_cnn_board'] = deepcopy(observed.get('raw_cnn_board'))
    raw['formula'] = dict(valid=formula.get('valid', False), product=formula.get('product'))


class CaptureEx:
    def __init__(self) -> None:
        self.boards: list = [None, None]
        self.queues: list = [(None, None), (None, None)]

    def capture(self, result: Any, frame: Any, row: dict) -> dict:
        from .live_v49f_capture import raw_side
        packet = row.get('_v49f')
        sides = packet['raw'] if packet is not None else [raw_side(s) for s in (result.p1, result.p2)]
        raw, tape = [], tape_next_enabled()
        for index, side in enumerate(deepcopy(sides)):
            add_fire_inputs(side, row, index)
            board = ((side.get('confirmed_board') or {}).get('board') or {}).get('array')
            if side['state']['state'] == 'STABLE' and board is not None and board != self.boards[index]:
                self.boards[index] = board
                self.queues[index] = (side.get('next_pair'), side.get('dnext_pair'))
            if tape:
                # 10-29: 持ち越しの NEXT は飛び・誤読でテープが作り直され、受け側の観測列が 4 組未満になる
                # (24l で相手が連鎖中の 36.5%)。テープには毎コマの値を別の鍵で渡し、確定はテープ自身が行う。
                side.update(tape_next_pair=side.get('next_pair'), tape_dnext_pair=side.get('dnext_pair'))
            raw.append(dict(side, next_pair=self.queues[index][0], dnext_pair=self.queues[index][1]))
        return dict(t=frame.media_sec, game=row['game_idx'], raw=raw)


def capture(owner: Any, result: Any, frame: Any, row: dict, epoch: Any) -> None:
    # 正式OBS抽出器のStableNextはengineのepoch切替をまたいで保持する。
    # 既観測のNEXTだけを保持し、新しい所有者で運転を始めたときに初期化する。
    if not hasattr(owner, '_ex_capture'):
        owner._ex_capture = CaptureEx()
    owner._ex_epoch = epoch
    row['_ex'] = owner._ex_capture.capture(result, frame, row)
