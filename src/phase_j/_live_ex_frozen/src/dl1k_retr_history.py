"""保持前のNEXTを確認して並べ、同じ試合の過去だけで位置を絞る。"""
from __future__ import annotations

import os
from typing import Any

ENV = 'PUYO_RETR_TSUMO_SYNC_HISTORY'


def enabled() -> bool:
    return os.environ.get(ENV, '0') == '1'


def observe(engine: Any, raw: list[dict]) -> None:
    """既存の確認時間・飛び時のrestartを使い、生NEXT専用の履歴を保つ。"""
    if not enabled():
        return
    from .exchange_tsumo_tape import TsumoTape
    from .dl1k_retr_tsumo import pair, FIELDS
    if getattr(engine, 'retr_history_game', None) != engine.state.game_idx:
        engine.retr_history_game = engine.state.game_idx
        engine.retr_history_tape = TsumoTape()
    for side, row in enumerate(raw):
        pairs = [pair(row.get(key)) for key in FIELDS]
        engine.retr_history_tape.observe(side, engine.state.t, *pairs,
                                        motion=bool(row.get('next_slide_motion', False)))


def candidates(engine: Any, side: int, seed: int, current: list, fallback: list) -> tuple[list, int]:
    """途中の欠測を手数で埋めず、最後に連続して確認できた並びを照合する。"""
    from .dl1k_retr_tsumo import pair
    if getattr(engine, 'retr_history_game', None) != engine.state.game_idx:
        return fallback, 0
    state = engine.retr_history_tape.sides[side]
    history = tuple(map(pair, state.own))
    if len(history) < 2 or tuple(current) != history[-2:]:
        return fallback, 0
    colors = tuple(map(pair, engine.knowledge.lookup.table.abs_colors[seed]))
    size = len(history)
    windows = list(zip(history, history[1:]))
    windows = [w for i, w in enumerate(windows) if i == 0 or w != windows[i-1]]
    positions = [i for i in fallback if matches_windows(colors, windows, i)]
    return positions, size


def matches_windows(colors: tuple, windows: list, position: int) -> bool:
    """同一のNEXT2組が続く間だけ、観測で区別不能な繰上がりを許す。"""
    current = windows[-1]
    for previous in reversed(windows[:-1]):
        position -= 1
        while position >= 0 and colors[position:position+2] == current:
            position -= 1
        if position < 0 or colors[position:position+2] != previous:
            return False
        current = previous
    return True


def recover(engine: Any, side: int, busy: bool, seed: int, depth: int) -> list | None:
    """leanの短い履歴や照合失敗も、生NEXT履歴が一意なら回復する。"""
    if not enabled():
        return None
    from .dl1k_retr_tsumo import synchronize
    start = synchronize(engine, side, busy, seed, -1, depth)
    if start < 0:
        return None
    table = engine.knowledge.lookup.table
    return [tuple(map(int, x)) for x in table.predict_absolute(seed, start, depth).tolist()]


def input_key(engine: Any) -> dict:
    tape = getattr(engine, 'retr_history_tape', None)
    return dict(game=getattr(engine, 'retr_history_game', None),
                sides=None if tape is None else [dict(own=s.own, queue=s.queue) for s in tape.sides])
