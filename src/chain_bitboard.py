"""高速連鎖判定エンジン (bitboard, numpy バッチ版)。

出典・帰属 (MIT License):
    本モジュールの消去判定式 (`_get_mask_pop_batch` の m3/m2 定式) および
    重力コンパクション (PEXT 相当) のアルゴリズムは、citrus610/ama
    (https://github.com/citrus610/ama, MIT License, Copyright (c) 2023 citrus610)
    の `core/fieldbit.cpp` / `core/field.cpp` を Python/numpy へ移植したもの。
    原著作者のアルゴリズム設計 (bitboard シフト+AND/OR による連結判定、
    PEXT によるコンパクション) に対する帰属を明記する。
    移植は本プロジェクト向けにゼロから numpy 実装し、SIMD (__m128i) 命令列は
    使用していない (numpy のベクトル演算に置換)。

目的:
    `src/chain.py` の `ChainSimulator` (BFS フラッドフィル方式) と等価な
    連鎖判定を、候補数 N に対してバッチ化した numpy 演算で高速に行う。
    探索の骨格 (発火せず積むビームサーチ等) は `src/indicators_v2.py` の
    既存設計をそのまま使う。本モジュールは「連鎖が何連鎖・何個消えるか」を
    判定するエンジン部分のみを高速化対象とする。

**重要 (backward compat / stateless 原則)**:
    `src/chain.py` の `ChainSimulator` は一切変更しない。既存指標・
    学習済み重み・既存 cycle の挙動には影響しない (完全に独立した新モジュール)。
    全関数 stateless (引数の配列を破壊しない)。

盤面ビット表現:
    1 列 = 1 個の uint16 (下位 13 bit を使用)。
    bit i は物理行 (BOARD_ROWS - 1 - i) に対応する
    (bit0 = 最下段 row=12、bit12 = 最上段(隠し段) row=0)。
    この向きにより「ビットを LSB 方向へ詰める」operationが
    「重力で下に詰める」ことと自然に対応する (ama の PEXT コンパクションと同じ発想)。

    我々の盤面は 13 行すべてが連結判定に参加する (`src/chain.py` の
    `find_groups` が row0(隠し段) も含めて flood fill する仕様、
    `docs/PUYO_RULES_CONFIRMED_2026-07-22.md` 参照)。
    ama 原典の `get_mask_12` (13bit中12bitのみ有効、最上段1bitを消去判定から
    除外) は ama 固有の「14段目バッファ行」概念に基づくもので、
    我々の盤面には存在しない。よって本移植では 13bit 全体
    (`FULL_MASK_13BIT`) をそのまま連結判定に使う (12bit制限は採用しない)。
    この差異は正当性回帰テスト (tests/test_chain_bitboard.py) で担保する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    Board,
)
from src.chain import MIN_ERASE_COUNT

# ============================
# 定数
# ============================

# 盤面は 13 行 (BOARD_ROWS) 全体を使う。ama の 12bit 制限は採用しない (上記docstring参照)。
FULL_MASK_13BIT: int = (1 << BOARD_ROWS) - 1  # = 0x1FFF = 8191

# 連結判定対象の色 (お邪魔・空・UNKNOWN を除く5色)。
TRACKED_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)

# 連鎖ループの安全弁 (ama の avec<Field,19> と同じ上限。理論上級者実用上限 ~19連鎖)。
MAX_CHAIN_STEPS: int = 19

_UINT16 = np.uint16
_MASK = _UINT16(FULL_MASK_13BIT)

# 16bit値 -> 立っているビット数 の事前計算済みルックアップテーブル (65536 要素、numpy化)。
# popcount をバッチ全体でベクトル化するために使う (Python ループ回避)。
_POPCOUNT_TABLE_16BIT: np.ndarray = np.array(
    [bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8,
)


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class BitboardChainResult:
    """バッチ内 1 盤面分の連鎖シミュレーション結果 (ChainResult の軽量版)。

    Attributes:
        chain_count: 総連鎖数。
        total_erased: 通常ぷよ消去数合計 (お邪魔除く)。
        total_ojama: お邪魔消去数合計。
        final_planes: 連鎖終了後の色別ビットボード (dict[color, uint16配列(6,)])。
    """
    chain_count: int
    total_erased: int
    total_ojama: int
    final_planes: "dict[int, np.ndarray]"


# ============================
# Board <-> ビットボード 変換
# ============================


def _pack_grids_to_planes(grids: np.ndarray) -> "dict[int, np.ndarray]":
    """grids (shape=(N, BOARD_ROWS, BOARD_COLS)) を一括ベクトル化してビット化する。

    Python ループは BOARD_ROWS (=13) 回のみ (バッチサイズ N に非依存)。
    `board_to_planes`/`batch_from_boards` の内部高速パス。

    Args:
        grids: 色コード整数配列、shape=(N, 13, 6)。

    Returns:
        dict[int, np.ndarray]: color -> shape=(N, 6) uint16 配列。
    """
    n = grids.shape[0]
    planes: "dict[int, np.ndarray]" = {}
    for color in (*TRACKED_COLORS, COLOR_OJAMA):
        col_plane = np.zeros((n, BOARD_COLS), dtype=_UINT16)
        mask = grids == color  # shape (N, 13, 6) bool
        for row in range(BOARD_ROWS):
            bit_index = BOARD_ROWS - 1 - row
            col_plane |= (mask[:, row, :].astype(_UINT16) << _UINT16(bit_index))
        planes[color] = col_plane
    return planes


def board_to_planes(board: Board) -> "dict[int, np.ndarray]":
    """Board を色別ビットボード (dict[color, uint16配列 shape=(6,)]) に変換する。

    内部で `_pack_grids_to_planes` のバッチ経路 (N=1) を使い、変換ロジックを
    1本化する (単発呼び出し用の薄いラッパー)。

    Args:
        board: 変換対象の盤面 (破壊しない)。

    Returns:
        dict[int, np.ndarray]: TRACKED_COLORS + COLOR_OJAMA をキーとする
        6要素 uint16 配列 (列ごとのビットマスク)。
    """
    grids = board._grid[np.newaxis, :, :]  # shape (1, 13, 6)
    batch = _pack_grids_to_planes(grids)
    return {color: arr[0] for color, arr in batch.items()}


def planes_to_board(planes: "dict[int, np.ndarray]") -> Board:
    """色別ビットボードを Board へ復元する。"""
    board = Board()
    for color, col_plane in planes.items():
        for col in range(BOARD_COLS):
            bits = int(col_plane[col])
            for bit_index in range(BOARD_ROWS):
                if bits & (1 << bit_index):
                    row = BOARD_ROWS - 1 - bit_index
                    board.set(row, col, color)
    return board


def batch_from_boards(boards: "list[Board]") -> "dict[int, np.ndarray]":
    """複数盤面をバッチ化する: dict[color, uint16配列 shape=(N, 6)]。

    `_pack_grids_to_planes` のベクトル化経路を使う (grid を stack して
    1回のバッチ変換で済ませる。N 件の Python ループを排除)。
    """
    if not boards:
        return {
            color: np.zeros((0, BOARD_COLS), dtype=_UINT16)
            for color in (*TRACKED_COLORS, COLOR_OJAMA)
        }
    grids = np.stack([b._grid for b in boards], axis=0)  # shape (N, 13, 6)
    return _pack_grids_to_planes(grids)


# ============================
# シフト演算 (ama の get_expand / get_mask_pop 相当、numpy バッチ版)
# ============================


def _shift_vertical(plane: np.ndarray, toward_lsb: bool) -> np.ndarray:
    """縦方向 (盤面内の上下) シフト。plane: shape (..., 6) uint16。

    ama の `_mm_srli_epi16` / `_mm_slli_epi16` (ビット単位シフト) に相当。
    """
    if toward_lsb:
        return (plane >> _UINT16(1)) & _MASK
    return (plane << _UINT16(1)) & _MASK


def _shift_horizontal(plane: np.ndarray, toward_low_col: bool) -> np.ndarray:
    """横方向 (列方向) シフト。plane: shape (..., 6) uint16。

    ama の `_mm_srli_si128(data, 2)` / `_mm_slli_si128(data, 2)`
    (16bit レーン=1列 単位のシフト、境界は0埋め) に相当。
    """
    result = np.zeros_like(plane)
    if toward_low_col:
        # 列 c <- 列 c+1 (右を見る)
        result[..., :-1] = plane[..., 1:]
    else:
        # 列 c <- 列 c-1 (左を見る)
        result[..., 1:] = plane[..., :-1]
    return result


def _expand(plane: np.ndarray) -> np.ndarray:
    """ama `get_expand`: 自身 | 上下左右シフト (1マス膨張、self含む)。"""
    return (
        plane
        | _shift_vertical(plane, toward_lsb=True)
        | _shift_vertical(plane, toward_lsb=False)
        | _shift_horizontal(plane, toward_low_col=True)
        | _shift_horizontal(plane, toward_low_col=False)
    )


def _get_mask_pop_batch(plane: np.ndarray) -> np.ndarray:
    """ama `get_mask_pop` の numpy バッチ移植 (1色分)。

    plane: shape (..., 6) uint16 (この色のビットボード、バッチ次元は任意)。

    Returns:
        np.ndarray: 同 shape。4連結以上に属するこの色のセルのビットが立つ。
    """
    m = plane & _MASK  # 我々は13bit全体を使う (ama の m12 相当を m13 に読替)

    u = _shift_vertical(m, toward_lsb=True) & m
    d = _shift_vertical(m, toward_lsb=False) & m
    r = _shift_horizontal(m, toward_low_col=True) & m
    l = _shift_horizontal(m, toward_low_col=False) & m

    ud_and = u & d
    lr_and = l & r
    ud_or = u | d
    lr_or = l | r

    m3 = (ud_and & lr_or) | (lr_and & ud_or)
    m2 = ud_and | lr_and | (ud_or & lr_or)

    m2_r = _shift_horizontal(m2, toward_low_col=True) & m2
    m2_l = _shift_horizontal(m2, toward_low_col=False) & m2
    m2_u = _shift_vertical(m2, toward_lsb=True) & m2
    m2_d = _shift_vertical(m2, toward_lsb=False) & m2

    core = m3 | m2_r | m2_l | m2_u | m2_d
    result = _expand(core) & m
    return result


# ============================
# PEXT 相当 (重力コンパクション、numpy バッチ版)
# ============================


def _pext_batch(values: np.ndarray, keep_masks: np.ndarray) -> np.ndarray:
    """PEXT (parallel bit extract) のバッチ版。

    values, keep_masks: 同 shape (uint16)。keep_masks のうち立っている bit
    位置の values ビットを、LSB 側へ順に詰めて返す
    (= ama `FieldBit::pop` の `pext16(v[i], ~v_mask[i])` に相当、
    「消えなかったセルを重力で下(LSB方向)へ詰める」操作)。

    O(BOARD_ROWS) の定数回ループ (バッチサイズ N に非依存) で計算する
    (13bit×13bit の全数ルックアップテーブルは 64M エントリで大きすぎるため、
    prefix-count 方式のビット分解ループを採用)。
    """
    result = np.zeros_like(values)
    out_pos = np.zeros_like(values)
    for bit in range(BOARD_ROWS):
        bit_flag = _UINT16(1 << bit)
        mask_bit_set = (keep_masks & bit_flag) != 0
        val_bit_set = (values & bit_flag) != 0
        contribute = np.where(val_bit_set & mask_bit_set, _UINT16(1), _UINT16(0))
        # out_pos だけ左シフトして result に OR する
        shifted = (contribute.astype(np.uint32) << out_pos.astype(np.uint32)).astype(_UINT16)
        result |= shifted
        out_pos = (out_pos + mask_bit_set.astype(out_pos.dtype)).astype(_UINT16)
    return result


# ============================
# バッチ連鎖シミュレーション本体
# ============================


def simulate_batch(
    planes: "dict[int, np.ndarray]",
) -> "list[BitboardChainResult]":
    """複数盤面を一括で連鎖シミュレートする。

    Args:
        planes: `batch_from_boards` で得た dict[color, (N,6)uint16配列]。

    Returns:
        list[BitboardChainResult]: バッチ内各盤面の結果 (順序保持)。
    """
    n = next(iter(planes.values())).shape[0]
    current = {color: arr.copy() for color, arr in planes.items()}
    ojama = current[COLOR_OJAMA]

    chain_count = np.zeros(n, dtype=np.int32)
    total_erased = np.zeros(n, dtype=np.int64)
    total_ojama = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)

    for _step in range(MAX_CHAIN_STEPS):
        if not active.any():
            break

        # 色ごとの pop mask を計算し OR で統合
        color_pop_masks = [_get_mask_pop_batch(current[c]) for c in TRACKED_COLORS]
        color_union = color_pop_masks[0]
        for m in color_pop_masks[1:]:
            color_union = color_union | m

        # has_pop: 各盤面で color_union が非ゼロ列を1つでも持つか (完全ベクトル化)。
        has_pop = np.any(color_union != 0, axis=-1)
        still_popping = active & has_pop
        if not still_popping.any():
            break

        # お邪魔隣接消去: color_union を1マス膨張し、お邪魔ビットとAND
        ojama_cleared = _expand(color_union) & ojama
        full_pop_mask = color_union | ojama_cleared

        # 消去数カウント (popcount ルックアップテーブルで完全ベクトル化、Pythonループ無し)。
        erased_per_board = _POPCOUNT_TABLE_16BIT[color_union].sum(axis=-1).astype(np.int64)
        ojama_per_board = _POPCOUNT_TABLE_16BIT[ojama_cleared].sum(axis=-1).astype(np.int64)
        total_erased += np.where(still_popping, erased_per_board, 0)
        total_ojama += np.where(still_popping, ojama_per_board, 0)
        chain_count += still_popping.astype(np.int32)

        # PEXT コンパクション: keep_mask = ~full_pop_mask (このステップで消えない盤面は keep_mask=全ビット1)
        keep_mask = (~full_pop_mask) & _MASK
        # まだ発火していない盤面はそのまま (keep_mask を全部1にして無変化にする)
        keep_mask = np.where(
            still_popping[:, None], keep_mask, np.full_like(keep_mask, FULL_MASK_13BIT),
        )

        for color in (*TRACKED_COLORS, COLOR_OJAMA):
            current[color] = _pext_batch(current[color], keep_mask)
        ojama = current[COLOR_OJAMA]

        active = still_popping  # 次ステップは今回発火した盤面のみ継続

    results: "list[BitboardChainResult]" = []
    for i in range(n):
        final_planes = {color: current[color][i].copy() for color in (*TRACKED_COLORS, COLOR_OJAMA)}
        results.append(BitboardChainResult(
            chain_count=int(chain_count[i]),
            total_erased=int(total_erased[i]),
            total_ojama=int(total_ojama[i]),
            final_planes=final_planes,
        ))
    return results


def simulate_single(board: Board) -> BitboardChainResult:
    """1 盤面のみを判定する薄いラッパー (バッチ API のテスト・単発呼び出し用)。"""
    batch = batch_from_boards([board])
    return simulate_batch(batch)[0]
