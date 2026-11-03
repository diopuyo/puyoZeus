"""盤面読取の一括 (ベクトル化) 経路 (既定 OFF、出力は従来経路とビット一致)。

背景: 配信の認識 1 コマのうち盤面読取は、2 盤面 × 72 マスを Python のループで
マスごとに切り出し・判定・分類しており、numpy / OpenCV / Rust 境界の呼出回数
(1 コマあたり数百回) が支配的だった。本モジュールは「呼出回数を減らすだけ」で、
**判定の式・分岐の順序・数値の演算順序は従来経路のまま**にする。

ビット一致のために守ること (崩すと認識結果が変わる):
- 浮動小数の総和・内積 (NCC の dot など) は束ねない。総和の順序が変わると最終桁が
  変わり、閾値ぎわで判定が反転しうる (np.einsum は dot と 76% のセルで不一致だった)。
  NCC は従来の `CellPatchFingerprint` / `is_empty_by_patch_fp` をマスごとに呼ぶ。
- 画素ごとに独立な整数演算 (cv2.cvtColor の BGR→HSV) は、パッチを並べた 1 枚へ
  まとめて変換しても各画素の値は同じ。
- マスごとの結果が他のマスに依存しない処理 (HSV 分類・ガード) だけを後ろへ寄せて
  1 回の Rust 呼出 (`puyo_core.classify_cells_hsv`) にまとめる。順序を入れ替えても
  各マスの分岐は従来と同じ式で評価する。
- 想定外の入力 (BackgroundFingerprint の距離型・静的背景マスク・uint8 以外・例外) は
  None を返して従来経路へ戻す。黙って別の結果を返さない。

検証: tests/test_live_board_vectorized.py と scripts/verify_live_board_vectorized.py
(実コマで ON/OFF の出力・内部記録・パイプライン結果をビット比較)。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)

# 盤面読取の対象セルは可視行 (HIDDEN_ROWS 以降) × 列。
VISIBLE_CELL_COUNT: int = (BOARD_ROWS - HIDDEN_ROWS) * BOARD_COLS
# read_board_hsv_only は元画像の寸法でクランプする従来の仕様を持つため、
# 1080p 以外の入力は従来経路へ戻す (寸法の取り違えを再現しない)。
SUPPORTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)
# 矩形表のキャッシュ上限。ROI 補正で region が少数種類ずつ増えるだけだが、
# 無限に増えないよう上限を超えたら全消去する。
RECT_TABLE_CACHE_LIMIT: int = 512
# 色分類の戻り値で「ぷよ色ではない」とみなすもの (従来のガード判定と同じ集合)。
NON_PUYO_COLORS: tuple[int, int] = (COLOR_EMPTY, COLOR_UNKNOWN)
# 画像のチャンネル数 (BGR / HSV はどちらも 3)。CNN 入力は BGR + HSV の 6 チャンネル。
BGR_CHANNELS: int = 3
CNN_INPUT_CHANNELS: int = 2 * BGR_CHANNELS
# CNN 前処理の正規化の分母 (従来の `/ 255.0` と同じ値・同じ型)。
PIXEL_MAX: float = 255.0

# 診断用の通過件数 (出力には一切影響しない)。検証台が「一括経路が実際に使われたか」を
# 母数つきで示すために読む。reset_stats() で 0 に戻す。
STATS: Counter = Counter()


def reset_stats() -> None:
    STATS.clear()


CellRect = tuple[int, int, int, int]
_RECT_TABLES: dict[tuple[int, int, int, int, int, int], "CellRectTable"] = {}


class CellRectTable:
    """1 領域 (盤面) の可視セルのサンプル矩形。フレーム寸法でクランプ済み。"""

    __slots__ = ("positions", "rects")

    def __init__(self, positions: tuple[tuple[int, int], ...],
                 rects: tuple[CellRect, ...]) -> None:
        self.positions = positions  # (raw_row, col)。行優先で従来のループ順と同じ
        self.rects = rects          # (x1, y1, x2, y2)


def cell_rect_table(region: Any, img_w: int, img_h: int) -> CellRectTable:
    """read_board の「cell_sample_rect → フレーム内へクランプ」を領域ごとに 1 度だけ行う。

    クランプ式は従来の read_board / read_board_hsv_only と同一。矩形は
    (region の幾何, 画像寸法) の純関数なのでキャッシュしてよい。
    """
    key = (region.x, region.y, region.width, region.height, img_w, img_h)
    hit = _RECT_TABLES.get(key)
    if hit is not None:
        return hit
    positions: list[tuple[int, int]] = []
    rects: list[CellRect] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            positions.append((row, col))
            rects.append((x1, y1, x2, y2))
    table = CellRectTable(tuple(positions), tuple(rects))
    if len(_RECT_TABLES) >= RECT_TABLE_CACHE_LIMIT:
        _RECT_TABLES.clear()
    _RECT_TABLES[key] = table
    return table


def slice_patches(image: np.ndarray, table: CellRectTable) -> list[np.ndarray]:
    """矩形表どおりに切り出した view の一覧 (コピーしない)。"""
    return [image[y1:y2, x1:x2] for x1, y1, x2, y2 in table.rects]


# ----------------------------------------------------------------------------
# 盤面 HSV 画像 (hsv_full) の確保
# ----------------------------------------------------------------------------

# 回転して使うバッファ数。従来は毎コマ np.zeros で 6MB を新規確保しており、盤面 2 領域へ
# 書き込むたびに OS のページ初回アクセス (約 2,000 回) が約 1.3ms を占めていた。
# 使い回しで新規確保を無くす。直近のコマの HSV の view (_live_hsv_pixels) が
# 次のコマの書込みで壊れないよう、数コマ分を回す (従来は毎回別の配列だった)。
HSV_BUFFER_COUNT: int = 4
HsvRect = tuple[int, int, int, int]


def _hsv_union_rect(region: Any, width: int, height: int) -> HsvRect:
    """ImageReader._board_hsv_frame と同じ「セル全体 + 領域」を含む最小の矩形。"""
    first = region.cell_sample_rect(HIDDEN_ROWS, 0)
    last = region.cell_sample_rect(BOARD_ROWS - 1, BOARD_COLS - 1)
    x1 = max(0, min(first[0], last[0], region.x, width - 1))
    y1 = max(0, min(first[1], last[1], region.y, height - 1))
    x2 = max(x1 + 1, min(max(first[2], last[2], region.x + region.width), width))
    y2 = max(y1 + 1, min(max(first[3], last[3], region.y + region.height), height))
    return x1, y1, x2, y2


class HsvBufferPool:
    """盤面 2 領域だけ HSV 変換した画像 (他は 0) を、回転バッファへ作る。

    従来の `np.zeros` + 領域ごとの `cvtColor` と全画素が同じ値になる。矩形が前回と
    変わった (ROI 補正) ときは、前回書いた矩形のうち今回の矩形でないものを 0 に戻す。
    """

    def __init__(self) -> None:
        self._buffers: list[np.ndarray | None] = [None] * HSV_BUFFER_COUNT
        self._written: list[list[HsvRect]] = [[] for _ in range(HSV_BUFFER_COUNT)]
        self._cursor = 0

    def render(self, frame: np.ndarray, regions: tuple[Any, ...]) -> np.ndarray:
        height, width = frame.shape[:2]
        rects = [_hsv_union_rect(region, width, height) for region in regions]
        slot = self._cursor
        self._cursor = (slot + 1) % HSV_BUFFER_COUNT
        buffer = self._buffers[slot]
        if buffer is None or buffer.shape != frame.shape or buffer.dtype != frame.dtype:
            buffer = self._buffers[slot] = np.zeros(frame.shape, dtype=frame.dtype)
            self._written[slot] = []
        for x1, y1, x2, y2 in self._written[slot]:
            if (x1, y1, x2, y2) not in rects:
                buffer[y1:y2, x1:x2] = 0
        for x1, y1, x2, y2 in rects:
            target = buffer[y1:y2, x1:x2]
            out = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV, dst=target)
            if out is not target and not np.shares_memory(out, target):
                target[...] = out
        self._written[slot] = rects
        STATS["hsv_frame_pooled"] += 1
        return buffer


def board_hsv_frame(reader: Any, frame: np.ndarray, regions: tuple[Any, ...]) -> np.ndarray:
    """ImageReader._board_hsv_frame の一括経路 (回転バッファ)。"""
    pool = reader.__dict__.get("_vectorized_hsv_pool")
    if pool is None:
        pool = reader._vectorized_hsv_pool = HsvBufferPool()
    return pool.render(frame, regions)


# ----------------------------------------------------------------------------
# HSV 分類の一括 (Rust)
# ----------------------------------------------------------------------------

def native_batch_usable(hsv_clf: Any) -> bool:
    """この分類器の classify を Rust 一括に置き換えてよいか。

    ColorClassifier.classify が既定実装のままで、ネイティブ経路が有効、vote_mode でない
    (vote_mode は別アルゴリズムで移植対象外) ときだけ True。
    """
    from src.image_reader import ColorClassifier
    return (
        isinstance(hsv_clf, ColorClassifier)
        and type(hsv_clf).classify is ColorClassifier.classify
        and bool(getattr(hsv_clf, "_enable_native_hsv", False))
        and not getattr(hsv_clf, "_vote_mode", False)
    )


def _patches_are_bgr_uint8(patches: list[np.ndarray]) -> bool:
    return all(
        p.size == 0 or (p.dtype == np.uint8 and p.ndim == 3 and p.shape[2] == BGR_CHANNELS)
        for p in patches
    )


def _classify_native_batch(hsv_clf: Any, patches: list[np.ndarray]) -> list[int] | None:
    """パッチを横に並べた 1 枚へ 1 回の cvtColor + 1 回の Rust 呼出で分類する。

    従来は 1 パッチごとに cvtColor + Rust を呼んでいた (ColorClassifier._classify_native)。
    cvtColor は画素ごとに独立なので並べた画像でも各画素の HSV は同じ。Rust 側は
    矩形ごとに独立に統計を取る。size==0 のパッチは従来どおり COLOR_EMPTY。
    """
    import puyo_core

    live = [i for i, p in enumerate(patches) if p.size > 0]
    result = [COLOR_EMPTY] * len(patches)
    if not live:
        return result
    if not _patches_are_bgr_uint8(patches):
        return None
    canvas = np.zeros(
        (max(patches[i].shape[0] for i in live), sum(patches[i].shape[1] for i in live), BGR_CHANNELS),
        dtype=np.uint8,
    )
    rects = np.zeros((len(live), 4), dtype=np.int32)
    x = 0
    for k, i in enumerate(live):
        height, width = patches[i].shape[:2]
        canvas[:height, x:x + width] = patches[i]
        rects[k] = (x, 0, x + width, height)
        x += width
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    out = puyo_core.classify_cells_hsv(
        canvas, hsv, rects, hsv_clf._native_ranges(), hsv_clf._native_params(),
    )
    for k, i in enumerate(live):
        result[i] = int(out[k])
    return result


def classify_patches(hsv_clf: Any, patches: list[np.ndarray]) -> list[int]:
    """`[int(hsv_clf.classify(p)) for p in patches]` と同じ結果を返す。

    ネイティブ経路が使えれば 1 回の Rust 呼出にまとめる。使えない・失敗した場合は
    従来どおり 1 パッチずつ classify する (例外も従来と同じ形で出る)。
    """
    if patches and native_batch_usable(hsv_clf):
        try:
            batch = _classify_native_batch(hsv_clf, patches)
        except Exception:
            batch = None
        if batch is not None:
            STATS["classify_native_calls"] += 1
            STATS["classify_native_cells"] += len(patches)
            return batch
    STATS["classify_per_patch_cells"] += len(patches)
    return [int(hsv_clf.classify(p)) for p in patches]


def hsv_only_colors(
    hsv_clf: Any, frame: np.ndarray, region: Any,
    src_h: int, src_w: int,
) -> list[int] | None:
    """read_board_hsv_only の可視 72 セルを一括分類した色 (行優先)。使えなければ None。"""
    if (src_h, src_w) != SUPPORTED_FRAME_SHAPE or not native_batch_usable(hsv_clf):
        STATS["hsv_only_fallback"] += 1
        return None
    if frame.shape[:2] != SUPPORTED_FRAME_SHAPE:
        STATS["hsv_only_fallback"] += 1
        return None
    table = cell_rect_table(region, src_w, src_h)
    try:
        colors = _classify_native_batch(hsv_clf, slice_patches(frame, table))
    except Exception:
        colors = None
    STATS["hsv_only_batch" if colors is not None else "hsv_only_fallback"] += 1
    return colors


# ----------------------------------------------------------------------------
# read_board の 1st pass (背景 FP による空判定)
# ----------------------------------------------------------------------------

def _guard_colors(hsv_target: Any, patches: list[np.ndarray]) -> list[int]:
    """従来の `try: int(hsv_target.classify(patch)) except Exception: COLOR_EMPTY` を一括にしたもの。"""
    if not patches:
        return []
    try:
        return classify_patches(hsv_target, patches)
    except Exception:
        colors: list[int] = []
        for patch in patches:
            try:
                colors.append(int(hsv_target.classify(patch)))
            except Exception:
                colors.append(COLOR_EMPTY)
        return colors


def first_pass(
    reader: Any, frame: np.ndarray, region: Any, bg_fp: Any,
    hsv_full: np.ndarray | None, skip_tier1: bool, board: Board,
) -> list | None:
    """read_board の 1st pass を一括で行い、cells_to_classify を返す (使えなければ None)。

    返り値の形は従来のループが作るものと同じ
    (row, col, patch, cell_bg_distance, cell_hsv_patch) のリスト (行優先)。
    EMPTY と確定したセルは board.set(…, COLOR_EMPTY) 済み。
    対応するのは「背景 FP なし」と「PatchBackgroundFingerprint (静的背景マスクなし)」。
    """
    from src.background_fingerprint import PatchBackgroundFingerprint

    img_h, img_w = frame.shape[:2]
    table = cell_rect_table(region, img_w, img_h)
    if getattr(reader, "_live_native_read_board", False):
        # Rust 1 回呼出の経路 (既定 OFF)。使えない・失敗なら None が返り、下の従来の一括経路へ進む。
        from src.live_native_read_board import native_first_pass
        native_cells = native_first_pass(
            reader, frame, region, table, bg_fp, hsv_full, skip_tier1, board,
        )
        if native_cells is not None:
            return native_cells
    patches = slice_patches(frame, table)
    if any(p.size == 0 for p in patches):
        STATS["first_pass_fallback"] += 1
        return None
    if bg_fp is None or hsv_full is None:
        STATS["first_pass_without_background"] += 1
        return [(r, c, p, None, None) for (r, c), p in zip(table.positions, patches)]
    if (not isinstance(bg_fp, PatchBackgroundFingerprint)
            or reader._get_static_mask_for_region(region) is not None
            or hsv_full.shape[:2] != frame.shape[:2]):
        STATS["first_pass_fallback"] += 1
        return None
    hsv_patches = slice_patches(hsv_full, table)
    if any(p.size == 0 for p in hsv_patches):
        STATS["first_pass_fallback"] += 1
        return None
    STATS["first_pass_patch_fp"] += 1
    return _first_pass_patch_fp(reader, bg_fp, table, patches, hsv_patches, skip_tier1, board)


def _tier1_pending(
    reader: Any, bg_fp: Any, table: CellRectTable,
    hsv_patches: list[np.ndarray], skip_tier1: bool,
) -> list[int]:
    """tier1 が NCC で「空」と答えたセルの添字 (HSV ガードは未適用)。

    従来の `_is_empty_tier1` を raw_bgr_patch=None (ガード省略) で呼ぶので、
    NCC の計算・有効性判定は従来の関数そのもの。ガードだけを後で一括にする。
    """
    if skip_tier1:
        return []
    pending: list[int] = []
    for i, (row, col) in enumerate(table.positions):
        visible_row = row - HIDDEN_ROWS
        if reader._is_empty_tier1(
            bg_fp.cell_at_patch(visible_row, col), hsv_patches[i], None,
            visible_row, col, raw_bgr_patch=None,
        ):
            pending.append(i)
    return pending


def _apply_tier1_guard(
    reader: Any, pending: list[int], patches: list[np.ndarray],
) -> list[int]:
    """NCC 空判定のうち、HSV 単独分類でもぷよ色でないセルだけを残す (W13 根治 案2)。"""
    if not reader._enable_patch_fp_hsv_guard:
        return pending
    hsv_target = getattr(reader._classifier, "_hsv", reader._classifier)
    colors = _guard_colors(hsv_target, [patches[i] for i in pending])
    return [i for i, color in zip(pending, colors) if color in NON_PUYO_COLORS]


def _first_pass_patch_fp(
    reader: Any, bg_fp: Any, table: CellRectTable,
    patches: list[np.ndarray], hsv_patches: list[np.ndarray],
    skip_tier1: bool, board: Board,
) -> list:
    """PatchBackgroundFingerprint の tier1 / tier2 を従来と同じ順序の意味で一括評価する。"""
    from src.image_reader import _has_puyo_highlight, _median_hsv_3ch
    from src.background_fingerprint import CellFingerprint, is_empty_by_fp

    n = len(patches)
    pending = _tier1_pending(reader, bg_fp, table, hsv_patches, skip_tier1)
    STATS["tier1_ncc_empty_candidates"] += len(pending)
    tier1_empty = set(_apply_tier1_guard(reader, pending, patches))
    STATS["tier1_empty_after_guard"] += len(tier1_empty)
    if reader._use_highlight_override:
        tier1_empty = {i for i in tier1_empty if not _has_puyo_highlight(hsv_patches[i])}
    distances: list[float | None] = [None] * n
    tier2_pending: list[int] = []
    for i, (row, col) in enumerate(table.positions):
        if i in tier1_empty:
            continue
        bg_cell = bg_fp.cell_at(row - HIDDEN_ROWS, col)
        cur_fp = CellFingerprint(*_median_hsv_3ch(hsv_patches[i]))
        distances[i] = float(cur_fp.distance_to(bg_cell))
        if is_empty_by_fp(cur_fp, bg_cell, threshold=reader._bg_threshold):
            tier2_pending.append(i)
    hsv_target = getattr(reader._classifier, "_hsv", reader._classifier)
    tier2_colors = _guard_colors(hsv_target, [patches[i] for i in tier2_pending])
    empty = set(tier1_empty)
    empty.update(i for i, color in zip(tier2_pending, tier2_colors) if color in NON_PUYO_COLORS)
    STATS["tier2_candidates"] += len(tier2_pending)
    STATS["tier2_empty"] += len(empty) - len(tier1_empty)
    cells: list = []
    for i, (row, col) in enumerate(table.positions):
        if i in empty:
            board.set(row, col, COLOR_EMPTY)
        else:
            cells.append((row, col, patches[i], distances[i], hsv_patches[i]))
    return cells


# ----------------------------------------------------------------------------
# CNN 入力の前処理 (8x8 の BGR+HSV 6ch 正規化)
# ----------------------------------------------------------------------------

def preprocess_patches(
    patches: list[np.ndarray], height: int, width: int,
    circle_mask: np.ndarray | None,
) -> np.ndarray:
    """`np.stack([patch_to_tensor(p)[0] for p in patches])` と同じ値の (N, 6, H, W) float32。

    従来の CnnPatchClassifier._patch_to_tensor をパッチごとに呼ぶ代わりに、
    リサイズ (INTER_AREA、cv2 の制約でチャネル 4 超は束ねられない) だけをパッチごとに行い、
    HSV 変換・結合・/255・マスク・並べ替えは全パッチ分を 1 回で行う。どれも画素・要素ごとに
    独立な演算なので、各要素の値は従来と同じ。size==0 のパッチは従来どおりゼロ。
    """
    count = len(patches)
    STATS["cnn_preprocess_calls"] += 1
    STATS["cnn_preprocess_patches"] += count
    resized = np.zeros((count, height, width, BGR_CHANNELS), dtype=np.uint8)
    live = np.zeros(count, dtype=bool)
    for i, patch in enumerate(patches):
        if patch.size:
            cv2.resize(patch, (width, height), dst=resized[i], interpolation=cv2.INTER_AREA)
            live[i] = True
    hsv = cv2.cvtColor(resized.reshape(count * height, width, BGR_CHANNELS), cv2.COLOR_BGR2HSV)
    combined = np.empty((count, height, width, CNN_INPUT_CHANNELS), dtype=np.uint8)
    combined[..., :BGR_CHANNELS] = resized
    combined[..., BGR_CHANNELS:] = hsv.reshape(count, height, width, BGR_CHANNELS)
    normalized = combined.astype(np.float32) / PIXEL_MAX
    if circle_mask is not None:
        normalized = normalized * circle_mask[None, :, :, None]
    normalized[~live] = 0.0
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))
