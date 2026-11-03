"""ツモの種の特定と続きの生成 (ama評価v5)。明示呼出専用で本番既定OFF。

観測したツモ列 (両側を共通番号で結合した連続列) と100%一致する (種, 開始位置) が
全探索で1つだけのときに「特定」とする。照合は録画のツモテープと同じ見え方
(同じ組の3連続は2つに見える) で行い、色の割り当てが試合の4色と食い違う種は除く。
特定できた試合は、観測した列はそのまま、その先を種から生成した続きで埋める。
特定できない試合は従来どおり補完 (合成ツモ) を使い、対戦は実ツモの末尾で止める。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from src.tsumo_seed_generator import PAIR_COUNT, PROJECT_COLOR_OF_LETTER, WIKI_COLOR_LETTERS
from src.tsumo_seed_identifier import SeedTable

MIN_OBSERVED_PAIRS = 8       # 偶然の一致の期待個数が1を大きく下回る最小の観測数 (後述の理論値)
MAX_START_OFFSET = 16        # 共通番号の先頭と種の開始位置のずれの上限 (実測400試合で0〜8)
TAPE_LENGTH = 200            # 1側のツモ列の長さ (対戦128手 + 先読み35組に足りる長さ)
COLOR_COUNT = 5
ALL_COLORS = frozenset(range(1, COLOR_COUNT+1))
EXCLUDED_LETTER_COLUMN = 4   # 色の割り当ての5番目 = 除かれる色
SEED_CACHE = Path('logs/ama_eval_v5/seed_table_cache.npz')

STATUS_IDENTIFIED = 'identified'
STATUS_TOO_SHORT = 'too_short'
STATUS_AMBIGUOUS = 'ambiguous'
STATUS_NO_MATCH = 'no_match'
STATUS_OFFSET = 'offset_out_of_range'
STATUS_ORIENTATION = 'orientation_mismatch'
STATUS_BAD_TAPE = 'bad_tape'


@dataclass(frozen=True)
class Identification:
    status: str
    seed: int | None = None
    start: int | None = None        # 先頭の観測が種のテープ視点で何組目か
    first: int | None = None        # 先頭の観測の共通番号
    observed: int = 0               # 観測した組数 (両側を結合した連続列)
    candidates: int = 0             # 0誤一致の (種, 開始位置) の数

    @property
    def identified(self) -> bool:
        return self.status == STATUS_IDENTIFIED


def load_table(cache: Path = SEED_CACHE) -> SeedTable:
    cache.parent.mkdir(parents=True, exist_ok=True)
    return SeedTable.build(cache)


def merged_known(tape: dict) -> dict | None:
    """両側の観測を共通番号で結合する。色矛盾・欠落があれば None。"""
    counters = tape['counters']
    if None in counters or not all(len(t) >= 3 for t in tape['tapes']):
        return None
    known = {}
    for start, pairs in zip(counters, tape['tapes']):
        for i, pair in enumerate(pairs):
            if known.setdefault(start+i, tuple(pair)) != tuple(pair):
                return None
    if any(i not in known for i in range(min(counters), max(known)+1)):
        return None
    return known


def excluded_color(colors: list) -> int:
    missing = ALL_COLORS-set(colors)
    assert len(missing) == 1, colors
    return next(iter(missing))


def excluded_mask(table: SeedTable, colors: list) -> np.ndarray:
    """試合の4色と色の割り当てが一致する種 (除かれる色が一致) の真偽値。"""
    letters = np.array([PROJECT_COLOR_OF_LETTER[c] for c in WIKI_COLOR_LETTERS])
    return letters[table.cmap[:, EXCLUDED_LETTER_COLUMN]] == excluded_color(colors)


def zero_mismatch_hits(table: SeedTable, observed: list, mask: np.ndarray) -> list:
    hits = []
    for start in range(PAIR_COUNT-len(observed)+1):
        seeds = table.identify_tape_view(observed, start, 0).seeds
        hits += [(int(s), start) for s in seeds if mask[s]]
    return hits


def orientation_ok(table: SeedTable, seed: int, start: int, observed: list) -> bool:
    """色違いの組は、観測 (a,b) が表のプール順 (x,y) と逆順 (b,a)。録画の全件で成立した向き。"""
    for k, (a, b) in enumerate(observed):
        x, y = table.abs_colors[seed, table.tape_to_absolute_index(seed, start+k)]
        if a != b and (a, b) != (int(y), int(x)):
            return False
    return True


def identify(table: SeedTable, tape: dict, colors: list) -> Identification:
    known = merged_known(tape)
    if known is None:
        return Identification(STATUS_BAD_TAPE)
    first = min(tape['counters'])
    observed = [known[i] for i in range(first, max(known)+1)]
    if len(observed) < MIN_OBSERVED_PAIRS:
        return Identification(STATUS_TOO_SHORT, first=first, observed=len(observed))
    hits = zero_mismatch_hits(table, observed, excluded_mask(table, colors))
    base = dict(first=first, observed=len(observed), candidates=len(hits))
    if len(hits) != 1:
        return Identification(STATUS_NO_MATCH if not hits else STATUS_AMBIGUOUS, **base)
    seed, start = hits[0]
    if not 0 <= start-first <= MAX_START_OFFSET:
        return Identification(STATUS_OFFSET, seed, start, **base)
    if not orientation_ok(table, seed, start, observed):
        return Identification(STATUS_ORIENTATION, seed, start, **base)
    return Identification(STATUS_IDENTIFIED, seed, start, **base)


def seed_pair(table: SeedTable, seed: int, absolute: int) -> tuple:
    """絶対の組番号 (128以上は2周目としてループ) の組。向きは観測と同じ (y, x)。"""
    x, y = table.abs_colors[seed, absolute % PAIR_COUNT]
    return int(y), int(x)


def synthetic_pair(colors: list, offset: int) -> tuple:
    """従来のv4と同じ補完 (観測4色の順に2組を交互)。"""
    start = offset % 2*2
    return tuple(colors[start:start+2])


def build_tapes(tape: dict, colors: list, ident: Identification | None = None,
                table: SeedTable | None = None, length: int = TAPE_LENGTH) -> dict:
    """1側ごとの長さ length のツモ列。観測は保ち、その先を種の続き (特定時) か補完で埋める。

    absolute は各位置の絶対の組番号 (特定時のみ。観測部は種のテープ視点から換算)。
    """
    known = merged_known(tape)
    if known is None:
        raise ValueError('開始可視3組または共通番号欠測')
    end, counters = max(known)+1, tape['counters']
    use_seed = ident is not None and ident.identified
    absolute = {}
    if use_seed:
        for i in range(ident.first, end):
            absolute[i] = table.tape_to_absolute_index(ident.seed, ident.start+i-ident.first)
        last = absolute[end-1]
    for i in range(end, max(counters)+length):
        known[i] = seed_pair(table, ident.seed, last+i-end+1) if use_seed else synthetic_pair(colors, i-end)
        if use_seed:
            absolute[i] = last+i-end+1
    tapes = [[tuple(known[s+i]) for i in range(length)] for s in counters]
    indexes = [[absolute.get(s+i) for i in range(length)] for s in counters] if use_seed else None
    return dict(tapes=tapes, absolute=indexes, real_end=[end-s for s in counters],
                seeded=use_seed)


def lap_pairs(absolute: list | None, cursor: int, visible: int = 3) -> int:
    """cursor手目までに見えた組のうち、2周目 (絶対番号128以上) に入った組数。"""
    if not absolute:
        return 0
    return sum(1 for a in absolute[:cursor+visible] if a is not None and a >= PAIR_COUNT)
