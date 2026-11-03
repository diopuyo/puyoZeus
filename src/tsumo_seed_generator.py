"""ツモ (配ぷよ) 列の生成器: 初期種 (uint16, 65536 通り) → 4 色ルールの 128 組。

原典: Puyo Nexus wiki「Puyo Puyo!! 20th Anniversary/Upcoming Pair Randomizer」の説明
(LCG・色の割り当て・3 段シャッフル・3 色プール先頭 4 個のコピー) から自前で実装した。
citrus610 の gist は答え合わせ (出力の一致確認) にだけ使い、コードは写していない。

仕様 (wiki)
- 乱数: seed = seed * 0x5D588B65 + 0x00269EC3 (uint32)。出力は上位 16bit。mod m は (high16 * m) >> 16。
- 色の割り当て: RNG を 5 回 (残り 5,4,3,2,1 の mod)。eスポーツの 4 色ルールは割り当ての先頭 4 色を使い、
  5 番目の色が「除かれる色」。色の割り当ては seed で決まる (試合ごとに色が入れ替わるのはこのため)。
- プール: 3/4/5 色の各 256 個を i % 色数 で埋め、16/32/64 個の行で 3 段シャッフル。
- 3 色プールの先頭 4 個 (= 最初の 2 組) を 4/5 色プールの先頭へ上書き。→ 最初の 2 組は 3 色以内。
- 4 色ルールで使うのは 4 色プール (256 個 = 128 組)。2 周目以降の扱いは実データで確認する (別途)。

全 65536 種を numpy で同時に進める (種の軸をベクトル化)。1 回の全表生成は数秒。
"""
from __future__ import annotations

import numpy as np

SEED_COUNT = 1 << 16
LCG_MUL = 0x5D588B65
LCG_ADD = 0x00269EC3
U32_MASK = 0xFFFFFFFF
HIGH_SHIFT = 16
POOL_SIZE = 256
POOL_COLORS = (3, 4, 5)
FOUR_COLOR_INDEX = 1                      # POOL_COLORS 中の 4 色プールの位置
SHUFFLE_ROW_LENGTHS = (0x10, 0x20, 0x40)  # 3 段シャッフルの行の長さ
COLORMAP_SIZE = 5
COPIED_PUYOS = 4                          # 3 色プール先頭からコピーする個数 (= 2 組)
PAIR_COUNT = POOL_SIZE // 2               # 128 組
WIKI_COLOR_LETTERS = "RGBYP"              # wiki の色の並び (R,G,B,Y,P)
# このプロジェクトの色コード (1 赤 2 青 3 緑 4 黄 5 紫) への対応
PROJECT_COLOR_OF_LETTER = {"R": 1, "G": 3, "B": 2, "Y": 4, "P": 5}


class _Rng:
    """全種を同時に進める LCG。state は uint64 配列 (値は uint32 範囲)。"""

    def __init__(self, seeds: np.ndarray) -> None:
        self.state = seeds.astype(np.uint64)

    def next_high16(self) -> np.ndarray:
        self.state = (self.state * np.uint64(LCG_MUL) + np.uint64(LCG_ADD)) & np.uint64(U32_MASK)
        return self.state >> np.uint64(HIGH_SHIFT)

    def next_mod(self, modulo: int) -> np.ndarray:
        return ((self.next_high16() * np.uint64(modulo)) >> np.uint64(HIGH_SHIFT)).astype(np.int64)


def _color_maps(rng: _Rng, count: int) -> np.ndarray:
    """色の割り当て (count, 5) : 各行は 0..4 の元の色番号 (R,G,B,Y,P の添字) を、使う順に並べたもの。"""
    letters = np.tile(np.arange(COLORMAP_SIZE, dtype=np.int64), (count, 1))
    out = np.zeros((count, COLORMAP_SIZE), dtype=np.int64)
    rows = np.arange(count)
    for step in range(COLORMAP_SIZE):
        left = COLORMAP_SIZE - step
        pick = rng.next_mod(left)
        out[:, step] = letters[rows, pick]
        # 選んだ要素を取り除いて詰める
        for j in range(left - 1):
            shift = j >= pick
            letters[:, j] = np.where(shift, letters[:, j + 1], letters[:, j])
    return out


def _shuffle(rng: _Rng, colors: int, count: int) -> np.ndarray:
    """1 つのプールを 3 段シャッフル。戻り値 (count, 256)。"""
    pool = np.tile((np.arange(POOL_SIZE) % colors).astype(np.uint8), (count, 1))
    rows = np.arange(count)
    for length in SHUFFLE_ROW_LENGTHS:
        for row in range(POOL_SIZE // length - 1):
            for _ in range(length >> 1):
                i1 = row * length + rng.next_mod(length)
                i2 = (row + 1) * length + rng.next_mod(length)
                a, b = pool[rows, i1].copy(), pool[rows, i2].copy()
                pool[rows, i1], pool[rows, i2] = b, a
    return pool


def generate_tables(seeds: np.ndarray | None = None,
                    copy_three_color: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """seeds (既定: 全 65536 種) の (色の割り当て (n,5), 4 色プール (n,256) uint8) を返す。

    4 色プールは先頭 4 個を 3 色プールの先頭 4 個で上書き済み (= 最初の 2 組は 3 色以内)。
    プール値 0..3 は色の割り当ての添字 (実色 = colormap[seed, 値])。
    copy_three_color=False は上書きなしの比較用 (実データで補正の有無を検証するための対照。本番は True)。
    """
    if seeds is None:
        seeds = np.arange(SEED_COUNT)
    seeds = np.asarray(seeds)
    rng = _Rng(seeds)
    cmap = _color_maps(rng, len(seeds))
    pools = [_shuffle(rng, colors, len(seeds)) for colors in POOL_COLORS]
    four = pools[FOUR_COLOR_INDEX]
    if copy_three_color:
        four[:, :COPIED_PUYOS] = pools[0][:, :COPIED_PUYOS]
    return cmap, four


def pairs_of(pool: np.ndarray) -> np.ndarray:
    """(n,256) → (n,128,2) の組 (組内の順序はプールの並びのまま)。"""
    return pool.reshape(pool.shape[0], PAIR_COUNT, 2)


def generate_one(seed: int) -> tuple[str, list[tuple[int, int]]]:
    """1 つの種の (色の割り当て文字列 'RGBYP' の並べ替え, 128 組)。人間向けの確認用。"""
    cmap, four = generate_tables(np.array([seed]))
    letters = "".join(WIKI_COLOR_LETTERS[i] for i in cmap[0])
    return letters, [(int(a), int(b)) for a, b in pairs_of(four)[0]]


def absolute_colors(cmap: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """プール値 (n,128,2) → プロジェクト色コード (1..5) の配列 (cmap は (n,5))。"""
    to_project = np.array([PROJECT_COLOR_OF_LETTER[c] for c in WIKI_COLOR_LETTERS], dtype=np.uint8)
    actual = np.take_along_axis(cmap, pairs.reshape(len(cmap), -1).astype(np.int64), axis=1)
    return to_project[actual].reshape(pairs.shape)
