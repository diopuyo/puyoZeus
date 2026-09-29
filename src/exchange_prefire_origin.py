"""E34: 収集側の保存条件を予測用起点にだけ適用する。"""
from __future__ import annotations

from collections import Counter
from copy import copy
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.board import Board
from src.board_state_machine import BoardState
from src.match_range_gate import MatchRangeGate, build


@dataclass(frozen=True)
class Origin:
    """現在層とは共有しない、保存条件を通った確定盤面。"""

    t_sec: float
    board: Board
    queue: np.ndarray


def origin_match_gate(video_id: str, root: Path) -> MatchRangeGate:
    """収集側と同じ境界ファイルを読み、未適用を成功扱いしない。"""
    gate = build(video_id, root, True)
    if not gate.active:
        raise ValueError(f'E34の試合範囲が未取得: {video_id} の matches.tsv が必要')
    return gate


def recorded_match_gate(header: dict, root: Path) -> MatchRangeGate:
    """新規収集の固定範囲を優先し、旧形式では既存ファイルを参照する。"""
    saved = header.get('prefire_match_ranges')
    if saved is None:
        return origin_match_gate(header['video_id'], root)
    gate = MatchRangeGate(tuple(tuple(pair) for pair in saved['ranges']), saved['source'])
    if not gate.active:
        raise ValueError('E34の保存済み試合範囲が空')
    return gate


class PrefireOriginGuard:
    """認識側のW48b保持印を使い、連鎖途中・試合外の起点更新を抑止する。"""

    def __init__(self, gate: MatchRangeGate) -> None:
        if not gate.active:
            raise ValueError('E34には有効な既存試合範囲が必要')
        self.gate = gate
        self.history: list[list[Origin]] = [[], []]
        self.counts: Counter = Counter()
        self.audit: list[dict] = []

    def reset(self) -> None:
        """旧試合の盤面を次試合へ持ち越さず、監査件数だけを残す。"""
        self.history = [[], []]

    def observe(self, sides: tuple, stamp: float) -> None:
        """collectのSTABLE・試合範囲・連鎖保持の順に保存を制限する。"""
        for idx, side in enumerate(sides):
            hold = getattr(side, 'prefire_origin_hold', None)
            if hold is None:
                raise ValueError('E34のW48b保持印が未取得: 保持印付き固定入力の再収集が必要')
            if side.state != BoardState.STABLE or side.confirmed_board is None:
                continue
            if not self.gate.allows(stamp):
                self.counts['outside_match'] += 1
                continue
            if hold:
                self.counts['chain_active'] += 1
                continue
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            self.history[idx].append(Origin(stamp, side.confirmed_board.copy(), queue))
            self.counts['saved'] += 1

    def select(self, chain: Any, event: Any, idx: int, game: int) -> Any:
        """通知内の盤面を採らず、発火より前の最後の適格起点を複製する。"""
        saved = next((s for s in reversed(self.history[idx]) if s.t_sec < chain.trigger_sec), None)
        board = saved.board.copy() if saved is not None else None
        if is_dataclass(event):
            selected = replace(event, before_board=board)
        else:
            selected = copy(event) if event is not None else None
            if selected is not None:
                selected.before_board = board
        self.audit.append(dict(game=game, side=chain.side, chain_id=chain.chain_id,
            trigger_sec=chain.trigger_sec, origin_sec=saved.t_sec if saved else None,
            board=saved.board._grid.tolist() if saved else None))
        return selected

    def summary(self) -> dict:
        """適用母数と起点を保存し、後の残差比較に利用する。"""
        return dict(counts=dict(self.counts), match_gate=self.gate.counts(), origins=self.audit)
