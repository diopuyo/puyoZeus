"""ツモ種の特定器: 観測した先頭 n 組 (色付き) から、候補の初期種 (65536 通り) を絞る。

照合は 2 通り (どちらも組内の順序は無視 = 順序なしの組で比べる)。
- 色の入れ替え不変 (`identify`): 観測の色 → 種の色 の単射を全部試す。色の名前を使わない。
- 絶対色 (`identify_absolute`): seed が決める色の割り当て (RNG 5 回) まで込みで比べる。
  録画のツモテープで、観測の絶対色が種の予測と一致することを確認済み (結果は報告書)。
- 誤読の混じった組への耐性: 不一致の組が max_mismatch 組以内の種も候補に残す。
- 状態を持たない (表 = SeedTable は読み取り専用。先頭 5 組の索引は遅延生成のキャッシュ)。

表の大きさ: 組 128 x 65536 種 x 1 byte (順序なしの組の符号) = 8 MiB (入れ替え不変用) + 8 MiB (絶対色用)。
生成表 (色の割り当て 5 個 + プール 256 個) は約 18.5 MiB。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Hashable, Optional, Sequence

import numpy as np

from src.phase_j._live_ex_frozen.src.tsumo_seed_generator import (PAIR_COUNT, SEED_COUNT, absolute_colors, generate_tables,
                                      pairs_of)

SEED_COLORS = 4                 # 4 色ルール
NO_MATCH_CODE = 255             # 写せない観測の組 (どの種の組とも一致しない符号)
PREFILTER_PAIRS = 6             # 許容ありの粗い絞り込みに使う先頭の組数
HASH_MULTIPLIER = 0x100000001B3  # 64bit FNV 風の乗数 (組の列の同一視ハッシュ)
HASH_MASK = (1 << 64) - 1
MAX_OBSERVED_COLORS = 5         # 観測の色の種類の上限 (5 色目以降は誤読として写せない扱い)
PREFIX_INDEX_PAIRS = 5          # strict 照合の高速索引に使う先頭の組数 (5 組で候補は P50 25 件)
ABS_BASE = 8                    # 絶対色の組の符号の基数 (色コード 1..5 < 8)
TAPE_RUN_CAP = 2                # ツモテープが見える同じ組の連続の上限 (3 連続以上は NEXT の変化が無く 2 つに見える)
PAD_CODE = 255                  # テープ視点表の末尾の詰め物
CACHE_VERSION = 1

Pair = tuple[Hashable, Hashable]


def pair_code(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """順序なしの組の符号 (小さい色 * 4 + 大きい色)。"""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return (lo * SEED_COLORS + hi).astype(np.uint8)


def abs_pair_code(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """絶対色 (1..5) の順序なしの組の符号。"""
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    return (lo * ABS_BASE + hi).astype(np.uint8)


@dataclass(frozen=True)
class Match:
    """特定結果。seeds は候補種、mismatches は各候補の不一致組数 (最良の色対応で)。"""

    seeds: np.ndarray
    mismatches: np.ndarray

    @property
    def unique(self) -> bool:
        return len(self.seeds) == 1


class SeedTable:
    """全 65536 種の表 (読み取り専用)。"""

    def __init__(self, cmap: np.ndarray, pool: np.ndarray) -> None:
        self.pool = pool
        self.cmap = cmap
        self.pairs = pairs_of(pool)                                  # (N,128,2) 値 0..3
        self.codes = pair_code(self.pairs[..., 0], self.pairs[..., 1])  # (N,128)
        self.abs_colors = absolute_colors(cmap, self.pairs)          # (N,128,2) 色コード 1..5
        self.abs_codes = abs_pair_code(self.abs_colors[..., 0], self.abs_colors[..., 1])
        self._prefix_cache: dict[int, np.ndarray] = {}
        self._tape_codes: Optional[np.ndarray] = None
        self._tape_index: Optional[np.ndarray] = None

    @classmethod
    def build(cls, cache: Optional[Path] = None, copy_three_color: bool = True) -> "SeedTable":
        """全表を生成する (cache があれば npz に保存・再利用。生成は数秒)。"""
        if not copy_three_color:
            cache = None                      # 対照表はキャッシュしない (キャッシュは補正あり固定)
        if cache is not None and cache.exists():
            data = np.load(cache)
            if int(data["version"]) == CACHE_VERSION:
                return cls(data["cmap"], data["pool"])
        cmap, pool = generate_tables(copy_three_color=copy_three_color)
        if cache is not None:
            np.savez_compressed(cache, cmap=cmap, pool=pool, version=np.array(CACHE_VERSION))
        return cls(cmap, pool)

    def nbytes(self) -> int:
        """入れ替え不変用の組の符号表の大きさ (byte)。"""
        return int(self.codes.nbytes)

    # ---- 特定 (色の入れ替え不変) ----
    def identify(self, observed: Sequence[Pair], start: int = 0, max_mismatch: int = 0) -> Match:
        """observed (先頭 = 絶対 start 組目) に合う種を返す。色は入れ替えを許す。"""
        n = min(len(observed), PAIR_COUNT - start)
        if n <= 0:
            return Match(np.arange(SEED_COUNT), np.zeros(SEED_COUNT, dtype=np.int64))
        obs = _normalize(observed[:n])
        mappings = _color_mappings(obs.distinct)
        rows = np.arange(SEED_COUNT)
        if max_mismatch == 0 and n >= PREFIX_INDEX_PAIRS:
            rows = self._strict_rows(start, obs, mappings)
        window = self.codes[rows, start:start + n]
        best = np.full(len(rows), n + 1, dtype=np.int64)
        for mapping in mappings:
            _accumulate(best, window, _observed_codes(obs, mapping), max_mismatch)
        keep = np.nonzero(best <= max_mismatch)[0]
        order = np.argsort(best[keep], kind="stable")
        return Match(rows[keep][order], best[keep][order])

    def _strict_rows(self, start: int, obs: "_Observed", mappings: list[tuple[int, ...]]) -> np.ndarray:
        """先頭 PREFIX_INDEX_PAIRS 組の同一視ハッシュが一致する種 (strict 用の高速な前絞り)。"""
        keys = self._prefix_keys(start)
        queries = []
        for mapping in mappings:
            head = _observed_codes(obs, mapping)[:PREFIX_INDEX_PAIRS]
            if (head == NO_MATCH_CODE).any():
                continue
            h = 0
            for code in head:
                h = (h * HASH_MULTIPLIER + int(code) + 1) & HASH_MASK
            queries.append(h)
        if not queries:
            return np.zeros(0, dtype=np.int64)
        return np.nonzero(keys == np.uint64(min(queries)))[0]

    def _prefix_keys(self, start: int) -> np.ndarray:
        """(N,) 先頭 PREFIX_INDEX_PAIRS 組 (start 起点) の色入れ替え同一視ハッシュ。遅延生成。"""
        if start not in self._prefix_cache:
            self._prefix_cache[start] = self.class_keys(start, PREFIX_INDEX_PAIRS)[-1]
        return self._prefix_cache[start]

    def mismatches_of(self, seed: int, observed: Sequence[Pair], start: int = 0) -> int:
        """ある種に対する不一致組数 (最良の色対応)。観測が種の予測範囲を超える分は数えない。"""
        n = min(len(observed), PAIR_COUNT - start)
        if n <= 0:
            return 0
        obs = _normalize(observed[:n])
        window = self.codes[seed:seed + 1, start:start + n]
        best = n
        for mapping in _color_mappings(obs.distinct):
            best = min(best, int((window != _observed_codes(obs, mapping)).sum()))
        return best

    # ---- 特定 (絶対色) ----
    def identify_absolute(self, observed: Sequence[Pair], start: int = 0, max_mismatch: int = 0) -> Match:
        """観測の絶対色 (プロジェクト色コード 1..5) で照合する。色の入れ替えは許さない。"""
        n = min(len(observed), PAIR_COUNT - start)
        if n <= 0:
            return Match(np.arange(SEED_COUNT), np.zeros(SEED_COUNT, dtype=np.int64))
        arr = np.array(observed[:n], dtype=np.int64)
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        miss = (self.abs_codes[:, start:start + n] != codes).sum(axis=1)
        seeds = np.nonzero(miss <= max_mismatch)[0]
        order = np.argsort(miss[seeds], kind="stable")
        return Match(seeds[order], miss[seeds][order])

    # ---- テープ視点 (同じ組の 3 連続以上を 2 つに潰した列) ----
    @property
    def tape_codes(self) -> np.ndarray:
        """(N,128) 絶対色の組の符号を、同じ並びの組の連続を TAPE_RUN_CAP 個までに潰して左詰めにした表。
        共通ツモテープは NEXT の変化で手を数えるので、同じ組が 3 回以上続くと 2 回としてしか見えない
        (既知の限界。実データでは矛盾の主因だった)。照合はこの視点でも行える。"""
        if self._tape_codes is None:
            same = np.zeros(self.abs_colors.shape[:2], dtype=bool)
            same[:, 1:] = (self.abs_colors[:, 1:] == self.abs_colors[:, :-1]).all(axis=2)
            run = np.ones(same.shape, dtype=np.int64)
            for k in range(1, PAIR_COUNT):
                run[:, k] = np.where(same[:, k], run[:, k - 1] + 1, 1)
            keep = run <= TAPE_RUN_CAP
            order = np.argsort(~keep, axis=1, kind="stable")
            packed = np.take_along_axis(np.where(keep, self.abs_codes, PAD_CODE), order, axis=1)
            counts = keep.sum(axis=1)
            packed[np.arange(PAIR_COUNT)[None, :] >= counts[:, None]] = PAD_CODE
            self._tape_codes = packed.astype(np.uint8)
            self._tape_index = order.astype(np.uint8)   # 添字は 0..127
        return self._tape_codes

    def tape_to_absolute_index(self, seed: int, tape_index: int) -> int:
        """テープ視点の添字 → 絶対の組番号 (潰された組の分だけ後ろにずれる)。"""
        _ = self.tape_codes
        return int(self._tape_index[seed, tape_index])

    def identify_tape_view(self, observed: Sequence[Pair], start: int = 0, max_mismatch: int = 0) -> Match:
        """identify_absolute のテープ視点版 (start もテープ視点の添字)。同じ組の 3 連続以上があっても照合できる。"""
        n = min(len(observed), PAIR_COUNT - start)
        if n <= 0:
            return Match(np.arange(SEED_COUNT), np.zeros(SEED_COUNT, dtype=np.int64))
        arr = np.array(observed[:n], dtype=np.int64)
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        miss = (self.tape_codes[:, start:start + n] != codes).sum(axis=1)
        seeds = np.nonzero(miss <= max_mismatch)[0]
        order = np.argsort(miss[seeds], kind="stable")
        return Match(seeds[order], miss[seeds][order])

    def tape_mismatches_of(self, seed: int, observed: Sequence[Pair], start: int = 0) -> list[int]:
        """テープ視点で不一致になった観測の添字。"""
        n = min(len(observed), PAIR_COUNT - start)
        arr = np.array(observed[:n], dtype=np.int64).reshape(-1, 2)
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        return np.nonzero(self.tape_codes[seed, start:start + n] != codes)[0].tolist()

    def abs_mismatches_of(self, seed: int, observed: Sequence[Pair], start: int = 0) -> list[int]:
        """絶対色で不一致になった観測の添字 (0 起点)。"""
        n = min(len(observed), PAIR_COUNT - start)
        arr = np.array(observed[:n], dtype=np.int64).reshape(-1, 2)
        codes = abs_pair_code(arr[:, 0], arr[:, 1])
        return np.nonzero(self.abs_codes[seed, start:start + n] != codes)[0].tolist()

    def predict_absolute(self, seed: int, start: int, count: int) -> np.ndarray:
        """seed の絶対色 (色の割り当て込み) の組 (count,2)。pool 順 (組内の順序は観測と逆の可能性あり)。"""
        return self.abs_colors[seed, start:start + count]

    # ---- 一意になるまでの手数の分布 ----
    def class_keys(self, start: int = 0, length: int = PAIR_COUNT, absolute: bool = False) -> np.ndarray:
        """(length, N) uint64: 先頭 n 組 (start 起点) の同一視ハッシュ (n=1..length)。
        absolute=False は色の入れ替え同一視 (24 通りの最小)、True は絶対色 (入れ替えなし)。"""
        keys = np.zeros((length, SEED_COUNT), dtype=np.uint64)
        mult = np.uint64(HASH_MULTIPLIER)
        if absolute:
            h = np.zeros(SEED_COUNT, dtype=np.uint64)
            for n in range(length):
                h = h * mult + self.abs_codes[:, start + n].astype(np.uint64) + np.uint64(1)
                keys[n] = h
            return keys
        lut = _relabel_tables()
        hashes = np.zeros((len(lut), SEED_COUNT), dtype=np.uint64)
        for n in range(length):
            code = self.codes[:, start + n].astype(np.int64)
            for k, table in enumerate(lut):
                hashes[k] = hashes[k] * mult + table[code].astype(np.uint64) + np.uint64(1)
            keys[n] = hashes.min(axis=0)
        return keys


@dataclass(frozen=True)
class _Observed:
    ids: np.ndarray        # (n,2) 観測の色を出現順の番号 (0..) にしたもの
    distinct: int


def _normalize(observed: Sequence[Pair]) -> _Observed:
    seen: dict[Hashable, int] = {}
    ids = np.zeros((len(observed), 2), dtype=np.int64)
    for i, pair in enumerate(observed):
        for j, color in enumerate(pair):
            ids[i, j] = seen.setdefault(color, len(seen))
    return _Observed(ids, len(seen))


def _color_mappings(distinct: int) -> list[tuple[int, ...]]:
    """観測の色番号 (0..distinct-1) → 種の色 (0..3) の単射。写せない観測色は -1。"""
    if distinct > MAX_OBSERVED_COLORS:
        return []
    if distinct <= SEED_COLORS:
        return [tuple(p) for p in permutations(range(SEED_COLORS), distinct)]
    out = []
    for drop in range(distinct):                       # 5 色観測: 1 色は誤読として写さない
        for p in permutations(range(SEED_COLORS), SEED_COLORS):
            it = iter(p)
            out.append(tuple(-1 if c == drop else next(it) for c in range(distinct)))
    return out


def _observed_codes(obs: _Observed, mapping: tuple[int, ...]) -> np.ndarray:
    lut = np.array(mapping, dtype=np.int64)
    a, b = lut[obs.ids[:, 0]], lut[obs.ids[:, 1]]
    codes = pair_code(np.maximum(a, 0), np.maximum(b, 0))
    return np.where((a < 0) | (b < 0), NO_MATCH_CODE, codes).astype(np.uint8)


def _accumulate(best: np.ndarray, window: np.ndarray, codes: np.ndarray, max_mismatch: int) -> None:
    """この色対応での不一致組数を、粗い絞り込み → 全組の順で計算し best を更新する。"""
    head = min(len(codes), PREFILTER_PAIRS)
    rough = (window[:, :head] != codes[:head]).sum(axis=1)
    alive = np.nonzero(rough <= max_mismatch)[0]
    if len(alive) == 0:
        return
    full = (window[alive] != codes).sum(axis=1)
    best[alive] = np.minimum(best[alive], full)


def _relabel_tables() -> list[np.ndarray]:
    """色の入れ替え 24 通りそれぞれの、組の符号 (0..15) → 入れ替え後の符号の表。"""
    tables = []
    for perm in permutations(range(SEED_COLORS)):
        table = np.zeros(SEED_COLORS * SEED_COLORS, dtype=np.uint8)
        for a in range(SEED_COLORS):
            for b in range(SEED_COLORS):
                lo, hi = sorted((perm[a], perm[b]))
                table[min(a, b) * SEED_COLORS + max(a, b)] = lo * SEED_COLORS + hi
        tables.append(table)
    return tables


def unique_hand_distribution(table: SeedTable, start: int = 0, length: int = PAIR_COUNT,
                             absolute: bool = False) -> dict:
    """各種が「先頭 n 組で他と区別できる (一意)」最小の n の分布と、n ごとの候補数 (全 65536 種)。"""
    keys = table.class_keys(start, length, absolute)
    first_unique = np.full(SEED_COUNT, 0, dtype=np.int64)       # 0 = length 組でも一意にならない
    per_n = []
    for n in range(length):
        _, inverse, counts = np.unique(keys[n], return_inverse=True, return_counts=True)
        size = counts[inverse]
        newly = (size == 1) & (first_unique == 0)
        first_unique[newly] = n + 1
        per_n.append(dict(n=n + 1, p50=float(np.percentile(size, 50)), p95=float(np.percentile(size, 95)),
                          max=int(size.max()), unique_seeds=int((size == 1).sum())))
    solved = first_unique[first_unique > 0]
    return dict(start=start, length=length, seeds=SEED_COUNT, absolute=absolute,
                never_unique=int((first_unique == 0).sum()),
                hands_to_unique=dict(p50=float(np.percentile(solved, 50)), p95=float(np.percentile(solved, 95)),
                                     max=int(solved.max()), min=int(solved.min())) if len(solved) else None,
                per_n=per_n)
