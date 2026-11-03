"""到着済みNEXT履歴・共通手番・整合した種候補だけでテープを狭める。"""
from __future__ import annotations

from typing import Any
import numpy as np
from src.phase_j._live_ex_frozen.src.exchange_tsumo_tape import TsumoTape
from src.phase_j._live_ex_frozen.src.tsumo_seed_identifier import SeedTable, abs_pair_code
from src.phase_j._live_ex_frozen.src.prefire_search_v410 import MAX_HANDS, VISIBLE_PAIRS, SEED, SIDE_SEED_STRIDE

MIN_OBSERVED = 8
PREFIX_PAIRS = 4
CODE_BITS = 6
PAIR_COUNT = 128
SIDES = (0, 1)


class SeedLookup:
    """既存の種表を索引化。新しい観測列だけを照合する。"""

    def __init__(self, table: SeedTable) -> None:
        self.table = table
        codes = table.tape_codes.astype(np.uint32)
        width = PAIR_COUNT-PREFIX_PAIRS+1
        keys = sum(codes[:, i:i+width] << (CODE_BITS*i) for i in range(PREFIX_PAIRS))
        self.width = width
        self.order = np.argsort(keys.ravel(), kind='stable')
        self.keys = keys.ravel()[self.order]
        self.last: dict[int, tuple] = {}

    def matches(self, side: int, own: list) -> list[tuple[int, int]]:
        key = tuple(own)
        if side in self.last and self.last[side][0] == key:
            return self.last[side][1]
        result = self._match(own)
        self.last[side] = key, result
        return result

    def _match(self, own: list) -> list[tuple[int, int]]:
        if not MIN_OBSERVED <= len(own) <= PAIR_COUNT:
            return []
        arr = np.asarray(own)
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        key = sum(int(c) << (CODE_BITS*i) for i, c in enumerate(codes[:PREFIX_PAIRS]))
        left, right = np.searchsorted(self.keys, key, 'left'), np.searchsorted(self.keys, key, 'right')
        flat = self.order[left:right]
        seeds, starts = flat//self.width, flat % self.width
        keep = starts+len(codes) <= PAIR_COUNT
        seeds, starts = seeds[keep], starts[keep]
        for i, code in enumerate(codes[PREFIX_PAIRS:], PREFIX_PAIRS):
            keep = self.table.tape_codes[seeds, starts+i] == code
            seeds, starts = seeds[keep], starts[keep]
        return list(zip(map(int, seeds), map(int, starts)))


class Knowledge:
    """状態保持は外部ラッパー。探索には時点固定の辞書だけを渡す。"""

    def __init__(self, lookup: SeedLookup | None = None) -> None:
        self.lookup = lookup
        self.tape = TsumoTape()
        self.game: int | None = None
        self.evidence = 0.

    def observe(self, stamp: float, game: int, raw: list[dict]) -> None:
        if game != self.game:
            self.tape.reset()
            self.game = game
        for side, row in enumerate(raw):
            pair = lambda key: row[key].get('tuple') if isinstance(row.get(key), dict) else row.get(key)
            self.tape.observe(side, stamp, pair('next_pair'), pair('dnext_pair'), row.get('next_slide_motion', False))
        self.evidence = stamp

    def snapshot(self, ctx: Any) -> dict:
        states = self.tape.sides
        fresh = [s.queue is not None and s.queue == s.candidate for s in states]
        hits = ([self.lookup.matches(i, s.own) for i, s in enumerate(states)]
                if self.lookup and all(fresh) else [[], []])
        seeds = {s for s, _ in hits[0]} & {s for s, _ in hits[1]}
        sides = [self._side(ctx, i, fresh[i]) for i in SIDES]
        matched = [[(s, p) for s, p in h if s in seeds] for h in hits]
        if len(seeds) == 1 and all(len(m) == 1 and s['queue_matches'] for m, s in zip(matched, sides)):
            candidates = [dict(s) for s in sides]
            for side in SIDES:
                self._seed(candidates[side], side, matched[side][0], ctx.colors)
            # 片側だけ絶対手番へ切り替えると共通乱数の位置がずれる。
            if all(s['seed'] is not None for s in candidates):
                sides = candidates
        return dict(evidence_max=self.evidence, aligned=self.tape.aligned, sides=sides)

    def _side(self, ctx: Any, side: int, fresh: bool) -> dict:
        st = self.tape.sides[side]
        queue = tuple(ctx.queues[side])
        observed = tuple(c for p in st.queue for c in p) if st.queue else ()
        # 完走直後は古いtriggerが残る。キューに操作中組の枠があるかを使う。
        busy = len(queue) == len(observed) if observed else ctx.triggers[side] is not None
        matches = fresh and queue[(0 if busy else 2):] == observed
        known = list(self.tape.known_future(side)) if matches else [tuple(queue[i:i+2]) for i in range(0, len(queue), 2)]
        if matches and not busy:
            known.insert(0, self.tape.active_pair(side) or (0, 0))
        return dict(known=known, seed_pairs=[], seed=None, seed_start=None, queue_matches=matches,
                    index=st.abs_next()-(not busy), common=self.tape.aligned and matches,
                    own=list(st.own), active_restored=matches and not busy and self.tape.active_pair(side) is not None)

    def _seed(self, item: dict, side: int, match: tuple, colors: tuple) -> None:
        seed, start = match
        st = self.tape.sides[side]
        busy_offset = len(item['known'])-len(self.tape.known_future(side))
        slot = start+len(st.own)-2-busy_offset
        absolute = self.lookup.table.tape_to_absolute_index(seed, slot)
        predicted = self.lookup.table.predict_absolute(seed, absolute, MAX_HANDS+VISIBLE_PAIRS).tolist()
        if any(c not in colors for pair in predicted for c in pair):
            return
        if any(all(c > 0 for c in pair) and sorted(pair) != sorted(predicted[i])
               for i, pair in enumerate(item['known'][:len(predicted)])):
            return
        item.update(seed=seed, seed_start=start, index=absolute, common=True,
                    seed_pairs=predicted)


def sample_tape(info: dict, colors: tuple, sample: int, side: int) -> list[tuple[int, int]]:
    """共通手番では同じ乱数列。種予測の組内向きは未知として平均する。"""
    count = MAX_HANDS+VISIBLE_PAIRS
    offset = max(0, info['index']) if info['common'] else 0
    rng = np.random.default_rng(SEED+sample+(0 if info['common'] else SIDE_SEED_STRIDE*(side+1)))
    values = rng.choice(colors, size=(count+offset, 2)).astype(int)[offset:]
    flip = np.random.default_rng(SEED+sample+PAIR_COUNT).integers(0, 2, size=count+offset)[offset:]
    for i, pair in enumerate(info['seed_pairs'][:count]):
        values[i] = pair[::(-1 if flip[i] else 1)]
    for i, pair in enumerate(info['known'][:count]):
        for j, color in enumerate(pair):
            if color in colors:
                values[i, j] = color
    return [tuple(map(int, pair)) for pair in values]
