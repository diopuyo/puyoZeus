"""盤面スナップショットの品質判定 (学習データ入口フィルタ用、2026-08-08).

非試合画面 (対戦カード紹介の白背景演出 / 観戦待ち受けロビー / 順位表画面) で
盤面領域が誤認識され、 STABLE スナップショットとして npz に記録されている
問題への対処。 `scripts/collect_boards_lean.py` が `force_in_match=True` で
MatchStateDetector の判定を無効化しているため、 認識側では非試合画面が
素通りする (実測: 全 123 本・890,945 スナップショット中 7,797 件 = 0.875%、
実画面 4/4 で真陽性を確認)。

本モジュールは「その盤面が物理的にあり得るか」だけで判定する stateless な
純関数群であり、 認識パイプラインには一切依存しない (観測指標は stateless
実装という規約に従う)。 認識側の根治 (in_match 判定の強化) とは独立に、
既存 npz を作り直さずに学習データから幻盤面を除外するために使う。

## 判定の物理的根拠 (定数はシーン逆算でなく物理から導出)
実戦では盤面 (可視 12 段 × 6 列) がおじゃまでほぼ埋まった時点で窒息死し、
試合が終わって盤面はリセットされる。 よって **おじゃま比率が極端に高い
満杯盤面が STABLE として継続記録されること自体があり得ない**。
"""
from __future__ import annotations

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA

# 隠し段 (row0) を除いた可視段の開始行。 窒息判定と同じ扱い (DEATH_ROW=1)。
VISIBLE_ROW_LO: int = 1
# 可視セル総数 (12 段 × 6 列 = 72)。
VISIBLE_CELL_COUNT: int = (BOARD_ROWS - VISIBLE_ROW_LO) * BOARD_COLS
# 「盤面の大半が埋まっている」= 可視 72 セル中 48 セル (2/3) 以上。
# 実戦でここまで積み上がると窒息が目前で、 安定状態として長く続かない。
PHANTOM_MIN_NONEMPTY: int = 48
# 非空セルのうちおじゃまが占める比率の下限。 相手の連鎖でおじゃまが降り
# 切っても自分の色ぷよが土台に残るため、 実戦で 0.7 を超える盤面は
# 事実上すでに死んでいる。
PHANTOM_MIN_OJAMA_RATIO: float = 0.7


def phantom_board_mask(grids: np.ndarray) -> np.ndarray:
    """盤面配列に対し「幻盤面か」の bool マスクを返す.

    Args:
        grids: (n, BOARD_ROWS, BOARD_COLS) の盤面配列。
            (BOARD_ROWS, BOARD_COLS) 単体を渡した場合も受け付ける。

    Returns:
        shape (n,) の bool 配列。 True = 幻盤面 (非試合画面由来の疑い)。
        単体盤面を渡した場合も shape (1,) で返す。
    """
    arr = np.asarray(grids)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    visible = arr[:, VISIBLE_ROW_LO:, :]
    nonempty = (visible != COLOR_EMPTY).sum(axis=(1, 2))
    ojama = (visible == COLOR_OJAMA).sum(axis=(1, 2))
    # nonempty=0 のとき 0 除算を避ける (空盤面は幻ではない)。
    ratio = np.where(nonempty > 0, ojama / np.maximum(nonempty, 1), 0.0)
    return (nonempty >= PHANTOM_MIN_NONEMPTY) & (ratio >= PHANTOM_MIN_OJAMA_RATIO)


def is_phantom_board(grid: np.ndarray) -> bool:
    """単一盤面が幻盤面かを返す (phantom_board_mask の単体版)."""
    return bool(phantom_board_mask(grid)[0])


def count_phantom_boards(grids: np.ndarray) -> tuple[int, int]:
    """(幻盤面数, 総数) を返す. 混入率の報告に使う."""
    mask = phantom_board_mask(grids)
    return int(mask.sum()), int(mask.size)
