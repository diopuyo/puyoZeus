"""保存入力を先頭から再生し、その時点までのNEXT観測だけで種を特定する。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np

from scripts.exchange_review_io import jsonl
from src.exchange_tsumo_tape import TsumoTape
from src.tsumo_seed_identifier import SeedTable, abs_pair_code, PAIR_COUNT

MIN_PAIRS = 6


def updates(path: Path) -> Iterator[dict]:
    """tsumo_seed_extract_tapes と同じ通知・同じTsumoTapeを使う。型復元は必要列だけ。"""
    for row in jsonl(path):
        if row.get('kind') != 'update':
            continue
        args = row['args']['tuple']
        sides = args[0]['namespace']
        result = dict(t_sec=float(args[3]), game_idx=int(args[4]), sides=[])
        for key in ('p1', 'p2'):
            side = sides[key]['namespace']
            pair = lambda name: side.get(name, {}).get('tuple') if isinstance(side.get(name), dict) else side.get(name)
            board = side.get('confirmed_board')
            result['sides'].append(dict(next=pair('next_pair'), dnext=pair('dnext_pair'),
                slide=bool(side.get('next_slide_motion')), state=side['state']['state'],
                board=board['board']['array'] if board else None))
        yield result


def candidates(table: SeedTable, observed: list) -> list[tuple[int, int]]:
    """開始位置を未知として全128組を探索。部分列を勝手に先頭1と仮定しない。"""
    n = len(observed)
    if not MIN_PAIRS <= n <= PAIR_COUNT:
        return []
    arr = np.asarray(observed)
    codes = abs_pair_code(arr[:, 0], arr[:, 1])
    seeds, starts = np.where(table.tape_codes[:, :PAIR_COUNT - n + 1] == codes[0])
    for i in range(1, n):
        keep = table.tape_codes[seeds, starts + i] == codes[i]
        seeds, starts = seeds[keep], starts[keep]
    return list(zip(seeds.tolist(), starts.tolist()))


class PrefixSeeds:
    """未来の入力を読めない逐次カーソル。特定結果の検証も現在のprefixで行う。"""

    def __init__(self, path: Path, table: SeedTable) -> None:
        self.table, self.stream, self.tape = table, iter(updates(path)), TsumoTape()
        self.pending = next(self.stream, None)
        self.game, self.last_t, self.cache = None, None, {}

    def at(self, t_sec: float, game: int) -> dict:
        while self.pending is not None and self.pending['t_sec'] <= t_sec:
            row = self.pending
            if row['game_idx'] != self.game:
                self.tape.reset()
                self.game, self.cache = row['game_idx'], {}
            for i, side in enumerate(row['sides']):
                self.tape.observe(i, row['t_sec'], side['next'], side['dnext'], side['slide'])
            self.last_t = row['t_sec']
            self.pending = next(self.stream, None)
        if self.game != game:
            return dict(status='入力未到達', input_sec=None, seed=None)
        return self.identify()

    def identify(self) -> dict:
        options = []
        for side in self.tape.sides:
            key = tuple(side.own)
            if key not in self.cache:
                self.cache[key] = candidates(self.table, side.own)
            options.append(self.cache[key])
        common = set(s for s, _ in options[0]) & set(s for s, _ in options[1])
        result = dict(status='未特定', input_sec=self.last_t, seed=None,
                      observed=[len(s.own) for s in self.tape.sides], candidates=len(common))
        if len(common) != 1:
            return result
        seed = next(iter(common))
        starts = [[p for s, p in side if s == seed] for side in options]
        if any(len(s) != 1 for s in starts):
            return result | dict(status='開始位置不明')
        indices = [self.table.tape_to_absolute_index(seed, start[0] + len(side.own) - 2)
                   for start, side in zip(starts, self.tape.sides)]
        return result | dict(status='特定済み', seed=seed, next_indices=indices,
            starts=[s[0] for s in starts], tape=self.table.predict_absolute(seed, 0, PAIR_COUNT).tolist())


def extend_tapes(seed: dict, row: dict) -> dict:
    """影ログの操作中組の有無を維持し、NEXTの位置から128組列へ接続する。"""
    if seed.get('seed') is None:
        return seed
    data, indices = row['exchange_input'], [0, 0]
    attack = int(data['attacker'])
    for side, field in ((attack, 'known_attacker'), (1 - attack, 'known_receiver')):
        index = seed['next_indices'][side]
        if side != attack and row.get('evidence', {}).get('active_pair_included'):
            index -= 1
        observed = data[field]
        predicted = seed['tape'][index:index + len(observed)] if index >= 0 else []
        if not observed or len(predicted) != len(observed) or any(sorted(a) != sorted(b)
                for a, b in zip(observed, predicted)):
            return seed | dict(status='操作中組/テープ位置不一致', seed=None)
        indices[side] = index
    return seed | dict(indices=indices)
