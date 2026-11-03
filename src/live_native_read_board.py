"""read_board の 1st pass を Rust の 1 回呼出にする経路 (既定 OFF、判定は従来経路と一致させる)。

背景: `live_board_vectorized` の 1st pass (`_first_pass_patch_fp`) でも、盤面 1 枚あたり
NCC (72 セル)・HSV 中央値・ガード分類を Python のセル単位ループで回しており、2 盤面で
1 コマ 7ms 前後 (実コマ・配布 Python・1 スレッド) を使っていた。本モジュールは盤面 1 枚を
`puyo_core.read_board_first_pass` の 1 回の呼出で判定する。判定の式と分岐の順序は
`src/live_board_vectorized.py` の `_first_pass_patch_fp` のままで、Rust 側
(native/puyo_core/src/board_first_pass.rs) が同じ順序で評価する。

出力の一致 (検証: scripts/verify_live_native_read_board.py):
- 整数だけで決まる値 (HSV 中央値・bg 距離・ガード分類・ハイライト) は従来経路とビット一致する。
- NCC は numpy の `mean`/`std` と同じペアワイズ総和を再現する。ただし `np.dot` は OpenBLAS の
  CPU ごとのカーネルで総和の順序が変わるため、内積 2 つだけは最終桁がずれうる。判定
  (NCC >= 閾値) が反転するのは閾値から 1e-15 以内に限られ、検証台が差の最大と閾値からの
  最小の余裕を母数つきで記録する。

使えない場合 (拡張が無い・旧形式の背景・静的背景マスク・numpy 以外の入力・例外) は None を返し、
呼出元が従来経路 (一括 Python 版、さらに従来のループ) へ戻る。黙って別の結果は返さない。

同じ旗 (live_native_read_board) で、LiveImages (発火前画像の盤面読み) の 2 つの分類器 (中央値・投票) も
Rust 経路にする (`enable_for_live_images`)。投票方式は native/puyo_core/src/hsv_vote.rs で、整数だけなので
ビット一致 (検証: scripts/verify_live_native_read_board.py の vote モード)。
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.board import COLOR_EMPTY, HIDDEN_ROWS, Board
from src.live_board_vectorized import STATS, CellRectTable, native_batch_usable

# Rust 関数名。拡張側 (native/puyo_core/src/board_first_pass.rs) と一致させる。
NATIVE_FUNCTION_NAME: str = "read_board_first_pass"
# Rust 側の出力の状態コード (board_first_pass.rs の STATUS_*)。
STATUS_CLASSIFY: int = 0
STATUS_TIER1_EMPTY: int = 1
STATUS_TIER2_EMPTY: int = 2
# 背景フラグのビット (board_first_pass.rs の FLAG_*)。
BG_FLAG_TIER1_ENABLED: int = 1
BG_FLAG_VALID: int = 2
# HSV のチャンネル数。
HSV_CHANNELS: int = 3
# 背景データ・矩形配列のキャッシュ上限 (ROI 補正で形が増えても無限に増やさない)。
CACHE_LIMIT: int = 16
# 旗の環境変数 (src/phase_j/live_speed.py が live_native_read_board から導く名前と同じ。試験で一致を確かめる)。
FLAG_ENV: str = "PUYO_LIVE_NATIVE_READ_BOARD"
# 診断用: 直近の失敗の理由 (出力には影響しない)。fail-silent を避けるため STATS と併せて残す。
LAST_FALLBACK_REASON: dict[str, str] = {}

_AVAILABLE: bool | None = None


def native_available() -> bool:
    """拡張が読め、必要な関数を持つか (結果は 1 度だけ調べて覚える)。"""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import puyo_core
            _AVAILABLE = callable(getattr(puyo_core, NATIVE_FUNCTION_NAME, None))
        except (ImportError, OSError):
            _AVAILABLE = False
    return _AVAILABLE


def reset_availability_cache() -> None:
    """試験用: 拡張の有無の判定を調べ直す。"""
    global _AVAILABLE
    _AVAILABLE = None


class TableArrays:
    """矩形表 1 つぶんの numpy 配列 (毎コマ作り直さない)。table を強参照して id の再利用を防ぐ。"""

    __slots__ = ("table", "rects", "rows", "cols", "shape_key")

    def __init__(self, table: CellRectTable) -> None:
        self.table = table
        self.rects = np.ascontiguousarray(np.asarray(table.rects, dtype=np.int32))
        positions = np.asarray(table.positions, dtype=np.intp)
        self.rows = np.ascontiguousarray(positions[:, 0])
        self.cols = np.ascontiguousarray(positions[:, 1])
        sizes = np.stack([self.rects[:, 3] - self.rects[:, 1], self.rects[:, 2] - self.rects[:, 0]], axis=1)
        self.shape_key = sizes.tobytes()


class BgPack:
    """背景指紋 1 枚ぶんの事前計算 (Python の `_as_bg_for` が作る値をセル順に連結したもの)。"""

    __slots__ = ("bg", "shape_key", "centered", "selfdot", "std", "flags", "median")

    def __init__(self, bg: Any, shape_key: bytes, centered: np.ndarray, selfdot: np.ndarray,
                 std: np.ndarray, flags: np.ndarray, median: np.ndarray) -> None:
        self.bg = bg
        self.shape_key = shape_key
        self.centered = centered
        self.selfdot = selfdot
        self.std = std
        self.flags = flags
        self.median = median


def table_arrays(reader: Any, table: CellRectTable) -> TableArrays:
    """矩形表に対応する配列 (読取器ごとのキャッシュ)。"""
    cache: dict[int, TableArrays] = reader.__dict__.setdefault("_native_table_arrays", {})
    entry = cache.get(id(table))
    if entry is None or entry.table is not table:
        if len(cache) >= CACHE_LIMIT:
            cache.clear()
        entry = cache[id(table)] = TableArrays(table)
    return entry


def build_bg_pack(bg_fp: Any, arrays: TableArrays) -> BgPack:
    """各セルの背景パッチを、そのセルの現パッチの形に整形した値で並べる。

    `CellPatchFingerprint._as_bg_for` を使うので、従来経路が毎コマ引く値と同じ計算・同じキャッシュ。
    """
    from src.background_fingerprint import BG_PATCH_VALID_V_MIN
    table = arrays.table
    count = len(table.positions)
    parts: list[np.ndarray] = []
    selfdot = np.empty(count, dtype=np.float64)
    std = np.empty(count, dtype=np.float64)
    flags = np.zeros(count, dtype=np.uint8)
    median = np.zeros((count, HSV_CHANNELS), dtype=np.int32)
    for k, ((row, col), (x1, y1, x2, y2)) in enumerate(zip(table.positions, table.rects)):
        visible_row = row - HIDDEN_ROWS
        cell = bg_fp.cell_at_patch(visible_row, col)
        _shaped, valid, _flat, b_std, centered, b_selfdot = cell._as_bg_for((y2 - y1, x2 - x1, HSV_CHANNELS))
        parts.append(np.ascontiguousarray(centered, dtype=np.float64).ravel())
        selfdot[k], std[k] = b_selfdot, b_std
        # 従来の `_is_empty_tier1` は V 中央値が下限未満なら NCC を実行しない (`not (v < 下限)` で写す)。
        flags[k] = (0 if cell.v_median < BG_PATCH_VALID_V_MIN else BG_FLAG_TIER1_ENABLED) | (
            BG_FLAG_VALID if valid else 0)
        fingerprint = bg_fp.cell_at(visible_row, col)
        median[k] = (int(fingerprint.h), int(fingerprint.s), int(fingerprint.v))
    return BgPack(bg_fp, arrays.shape_key, np.ascontiguousarray(np.concatenate(parts)),
                  selfdot, std, flags, median)


def bg_pack(reader: Any, bg_fp: Any, arrays: TableArrays) -> BgPack:
    """背景指紋 + 矩形の形 に対応する事前計算 (読取器ごとのキャッシュ)。背景を取り直せば作り直す。"""
    cache: dict[tuple[int, bytes], BgPack] = reader.__dict__.setdefault("_native_bg_packs", {})
    key = (id(bg_fp), arrays.shape_key)
    entry = cache.get(key)
    if entry is None or entry.bg is not bg_fp:
        if len(cache) >= CACHE_LIMIT:
            cache.clear()
        entry = cache[key] = build_bg_pack(bg_fp, arrays)
        STATS["native_bg_pack_built"] += 1
    return entry


def first_pass_params(reader: Any, skip_tier1: bool) -> dict[str, Any]:
    """判定の定数と旗。値は Python 側の定数・設定を単一情報源にして Rust へ渡す。"""
    from src import background_fingerprint as bf
    threshold = reader._patch_ncc_threshold
    return dict(
        ncc_threshold=float(bf.PATCH_NCC_EMPTY_THRESHOLD if threshold is None else threshold),
        ncc_std_min=float(bf.PATCH_NCC_STD_MIN),
        ncc_uniform_fallback=float(bf.PATCH_NCC_UNIFORM_FALLBACK),
        bg_threshold=float(reader._bg_threshold),
        h_weight=float(bf.H_WEIGHT), s_weight=float(bf.S_WEIGHT), v_weight=float(bf.V_WEIGHT),
        skip_tier1=bool(skip_tier1),
        tier1_guard=bool(reader._enable_patch_fp_hsv_guard),
        highlight_override=bool(reader._use_highlight_override),
        highlight_v_min=float(bf.HIGHLIGHT_V_MIN), highlight_s_max=float(bf.HIGHLIGHT_S_MAX),
        highlight_y_top=float(bf.HIGHLIGHT_REGION_Y_RATIO),
        highlight_y_bottom=float(bf.HIGHLIGHT_REGION_Y_RATIO_LOW),
        highlight_min_ratio=float(bf.HIGHLIGHT_MIN_PIXEL_RATIO),
    )


def _is_uint8_color_image(image: Any) -> bool:
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3
            and image.shape[2] == HSV_CHANNELS)


def _images_usable(frame: Any, hsv_full: Any) -> bool:
    """frame は uint8 の 3 チャンネル、hsv は同じ形で C 連続 (Rust が平坦な列として読む)。"""
    return (_is_uint8_color_image(frame) and _is_uint8_color_image(hsv_full)
            and hsv_full.shape == frame.shape and hsv_full.flags.c_contiguous)


def unsupported_reason(reader: Any, frame: np.ndarray, bg_fp: Any, hsv_full: np.ndarray | None,
                       region: Any) -> str | None:
    """この入力を Rust 経路で扱えない理由 (扱えるなら None)。"""
    from src import background_fingerprint as bf
    from src.background_fingerprint import PatchBackgroundFingerprint
    if not native_available():
        return "native_unavailable"
    if not isinstance(bg_fp, PatchBackgroundFingerprint) or hsv_full is None:
        return "no_patch_background"
    if not bf.ENABLE_DIRECT_PEARSON_NCC:
        return "ncc_legacy_mode"
    if reader._get_static_mask_for_region(region) is not None:
        return "static_mask"
    if not _images_usable(frame, hsv_full):
        return "image_layout"
    hsv_target = getattr(reader._classifier, "_hsv", reader._classifier)
    if not native_batch_usable(hsv_target):
        return "hsv_classifier"
    return None


def call_native(reader: Any, frame: np.ndarray, hsv_full: np.ndarray, arrays: TableArrays,
                pack: BgPack, skip_tier1: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rust を 1 回呼ぶ。戻りは (状態, bg 距離, NCC)。"""
    import puyo_core
    hsv_target = getattr(reader._classifier, "_hsv", reader._classifier)
    return getattr(puyo_core, NATIVE_FUNCTION_NAME)(
        frame, hsv_full, arrays.rects, pack.centered, pack.selfdot, pack.std, pack.flags, pack.median,
        hsv_target._native_ranges(), hsv_target._native_params(),
        first_pass_params(reader, skip_tier1),
    )


def materialize(frame: np.ndarray, hsv_full: np.ndarray, arrays: TableArrays,
                status: np.ndarray, distance: np.ndarray, board: Board) -> list:
    """Rust の出力を従来経路と同じ形 (row, col, patch, bg 距離, hsv patch) の一覧にする。

    EMPTY と確定したセルは board へ COLOR_EMPTY を書く (従来経路と同じ)。
    """
    empty = status != STATUS_CLASSIFY
    if empty.any():
        board._grid[arrays.rows[empty], arrays.cols[empty]] = COLOR_EMPTY
    rects = arrays.rects
    cells: list = []
    for i in np.flatnonzero(~empty).tolist():
        x1, y1, x2, y2 = rects[i].tolist()
        cells.append((int(arrays.rows[i]), int(arrays.cols[i]), frame[y1:y2, x1:x2],
                      float(distance[i]), hsv_full[y1:y2, x1:x2]))
    return cells


def native_first_pass(
    reader: Any, frame: np.ndarray, region: Any, table: CellRectTable, bg_fp: Any,
    hsv_full: np.ndarray | None, skip_tier1: bool, board: Board,
) -> list | None:
    """`live_board_vectorized.first_pass` と同じ返り値 (cells_to_classify) を Rust 1 回で作る。

    使えない・失敗したら None (呼出元が従来経路へ戻る)。理由は STATS と LAST_FALLBACK_REASON に残る。
    """
    from src.phase_j.live_color_roi import native_image
    frame = native_image(frame, table.rects)
    reason = unsupported_reason(reader, frame, bg_fp, hsv_full, region)
    if reason is not None:
        STATS["native_first_pass_fallback_" + reason] += 1
        LAST_FALLBACK_REASON["reason"] = reason
        return None
    try:
        arrays = table_arrays(reader, table)
        pack = bg_pack(reader, bg_fp, arrays)
        status, distance, _ncc = call_native(reader, frame, hsv_full, arrays, pack, skip_tier1)
    except Exception as error:   # 拡張側の検査 (形・範囲) や不整合。従来経路へ戻して続行する
        STATS["native_first_pass_fallback_error"] += 1
        LAST_FALLBACK_REASON["reason"] = f"error: {type(error).__name__}: {error}"
        return None
    STATS["first_pass_native"] += 1
    return materialize(frame, hsv_full, arrays, status, distance, board)


def enable_for_live_images(classifiers: list[Any]) -> list[bool]:
    """LiveImages (発火前画像の盤面読み) の分類器を、旗が ON のときだけネイティブ (Rust) にする。

    中央値方式は既存のネイティブ HSV (ビット一致)、投票方式 (vote_mode) は hsv_vote.rs (整数でビット一致)。
    旗が OFF・拡張が無い環境では何も変えず、分類器ごとの結果 (有効にできたか) を返す。
    """
    if os.environ.get(FLAG_ENV) != "1":
        return [False] * len(classifiers)
    results: list[bool] = []
    for classifier in classifiers:
        enable = classifier.enable_native_vote if classifier._vote_mode else classifier.enable_native_hsv
        results.append(bool(enable(True)))
    return results
