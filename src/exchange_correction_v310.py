"""到着済みの盤面・NEXT・式だけで観測補正の混入を棄却する。"""
from __future__ import annotations

from typing import Any
from src.exchange_count_prediction import COLS, histogram, simulate, validate_times
from src.exchange_count_v34 import grounded

PAIR_SIZE = 2


def observation_gate(frame: dict, initial_next: list[int] | None, origins: list[dict],
                     formulas: list[dict], previous: list[dict], colors: list[int],
                     policy: dict, native: Any) -> tuple[list[int] | None, str]:
    """不確かなら補正しない。正解盤・連鎖終了時刻を引数に持たない。"""
    stamp, sources = frame['t'], frame['sources']
    validate_times(stamp, sources+formulas+previous)
    if not frame['normal_score'] or frame['state'] != 'STABLE':
        return None, '通常得点とSTABLEが未確認'
    if not formulas or stamp-formulas[-1]['t'] < policy['quiet_sec']:
        return None, '式表示の直後'
    if initial_next is None:
        return None, '次の組の混入またはNEXT未確認'
    if not sources or any(c not in (0, 9, *colors) for s in sources for c in s['board']):
        return None, '試合外色・演出'
    boards = [grounded(s['board']) for s in sources]
    board = boards[-1]
    if frame.get('next_changes', int(frame.get('next') != initial_next)):
        removed = [[a for a, b in zip(s['board'], settled) if a != b] for s, settled in zip(sources, boards)]
        if frame.get('next_changes', 1) != 1 or any(sorted(c) != sorted(initial_next) for c in removed):
            return None, '次の組の混入またはNEXT未確認'
    if any(b != board for b in boards) or any(sum(a != b for a, b in zip(s['board'], b)) > PAIR_SIZE
                                            for s, b in zip(sources, boards)):
        return None, '分類器不一致または落下・演出'
    if frame.get('board') != board or any(board[:COLS*2]) or simulate(board, native)['erased']:
        return None, '確定盤不一致・隠し段不明・消去中'
    counts = histogram(board)
    if not origins or counts[-1] > max(v['origin'].count(9) for v in origins):
        return None, '新しいおじゃまの混入'
    maximum = max(sum(c in colors for c in v['origin']) for v in origins)
    if sum(counts[:-1]) > maximum-sum(s['erased'] for s in formulas):
        return None, '次の組または未消去ぷよの混入'
    recent = [f for f in previous if f['t'] >= stamp-policy['stable_sec']]
    if not recent or stamp-recent[0]['t'] < policy['stable_sec']-policy['sample_tolerance']:
        return None, '連続観測不足'
    if any(f['state'] != 'STABLE' or f.get('board') != board or f.get('next') != frame.get('next') for f in recent):
        return None, '確定盤またはNEXTが移動中'
    return counts, '採用'
