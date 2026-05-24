"""W3.0: 隠し段 (row 0、13 段目) の量子的ぷよ位置推論。

ぷよぷよでは row 0 = 13 段目 = 画面外の隠し段で、回し入れで puyo が
入ることがある。本モジュールは:
    - prev_board → cur_board の差分でペア落下を検出
    - 観測された画面内セル数と「ペアは 2 個」のルールから
      隠し段に積まれた残りぷよの位置・色を確率分布で推論

入力:
    prev_board: t-1 の確定盤面
    cur_board: t の観測盤面
    prev_next_pair: t-1 時点の next pair (top, bot) - 落下したペアの色

出力:
    ProbabilisticBoard - 隠し段セルの確率分布が更新されたもの

推論ロジック:
    新規 = prev EMPTY → cur 非 EMPTY のセル
    Case A (新規 2 セル、ペア整合): 隠し段は確定 EMPTY
    Case B (新規 1 セル):
        - もう 1 セルは隠し段にある
        - 色は next_pair の残った方
        - 列は不明 (落下列の候補から確率分布)
    Case C (新規 0 セル): 両方が隠し段の可能性
        - ただし通常は連鎖アニメ中なのでスキップ推奨
    Case D (新規 3+ セル): 連鎖中などで判定不能、スキップ

Phase I: 自己教師あり learnt calibration (Platt scaling) を任意で適用.
    apply_calibration=True 時、`data/verify/hidden_row_calibration.json`
    があればそれを読み込み、各セル分布の missing_color 確率を
    `sigmoid(a * p + b)` で補正する (再正規化付き).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)
from src.probabilistic_board import ProbabilisticBoard, ProbabilisticCell

# 確定的でない色 (next_pair に含まれていたら推論スキップ)
SKIP_COLORS = frozenset({COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA})

# Calibration JSON のデフォルトパス (apply_calibration=True 時に読み込み)
DEFAULT_CALIBRATION_PATH: Path = Path("data/verify/hidden_row_calibration.json")


@dataclass(frozen=True)
class HiddenInferenceResult:
    """推論結果。"""
    n_new_cells: int            # prev → cur で新規出現したセル数
    cells_added_to_hidden: int  # 隠し段に追加した確定セル数
    cells_with_distribution: int  # 確率分布として推論されたセル数
    skipped_reason: str | None


def _collect_new_cells(
    prev: Board, cur: Board,
) -> list[tuple[int, int, int]]:
    """prev EMPTY → cur 非 EMPTY (UNKNOWN 以外) のセル一覧。"""
    out: list[tuple[int, int, int]] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            p = int(prev.get(row, col))
            c = int(cur.get(row, col))
            if p == COLOR_EMPTY and c not in (COLOR_EMPTY, COLOR_UNKNOWN):
                out.append((row, col, c))
    return out


def _column_top_row(board: Board, col: int) -> int:
    """指定列で最上部 (最も上にある) の非 EMPTY セルの row を返す。
    全 EMPTY の場合は BOARD_ROWS (= 13)、つまり「ぷよ無し」。
    """
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        if int(board.get(row, col)) != COLOR_EMPTY:
            return row
    return BOARD_ROWS


def infer_hidden_row(
    prev_board: Board,
    cur_board: Board,
    prev_next_pair: tuple[int, int] | None,
    *,
    apply_calibration: bool = False,
    calibration_path: Path | str | None = None,
) -> tuple[ProbabilisticBoard, HiddenInferenceResult]:
    """隠し段 (row 0..HIDDEN_ROWS-1) の確率分布を推論。

    可視領域は cur_board をそのまま反映。
    隠し段は推論結果で更新。

    Args:
        prev_board: 1 frame 前の確定盤面
        cur_board: 現フレームの観測盤面
        prev_next_pair: 1 frame 前の next pair (top, bot)
        apply_calibration: True で Phase I calibration JSON を後 hook 適用
        calibration_path: 上書き用パス (None で DEFAULT_CALIBRATION_PATH)

    Returns:
        (probabilistic_board, hidden_inference_result)
    """
    pboard, result = _infer_hidden_row_core(
        prev_board, cur_board, prev_next_pair,
    )
    if apply_calibration:
        _apply_calibration_inplace(pboard, calibration_path)
    return pboard, result


def _infer_hidden_row_core(
    prev_board: Board,
    cur_board: Board,
    prev_next_pair: tuple[int, int] | None,
) -> tuple[ProbabilisticBoard, HiddenInferenceResult]:
    """calibration なしの素のヒューリスティック推論本体."""
    pboard = ProbabilisticBoard.from_board(cur_board)

    if prev_next_pair is None:
        return pboard, HiddenInferenceResult(
            n_new_cells=0, cells_added_to_hidden=0,
            cells_with_distribution=0, skipped_reason="no_next_pair",
        )
    if any(c in SKIP_COLORS for c in prev_next_pair):
        return pboard, HiddenInferenceResult(
            n_new_cells=0, cells_added_to_hidden=0,
            cells_with_distribution=0,
            skipped_reason="next_pair_not_definite",
        )

    new_cells = _collect_new_cells(prev_board, cur_board)
    n = len(new_cells)

    if n == 2:
        # 通常落下、隠し段は EMPTY 確定
        for r in range(HIDDEN_ROWS):
            for c in range(BOARD_COLS):
                pboard.set_certain(r, c, COLOR_EMPTY)
        return pboard, HiddenInferenceResult(
            n_new_cells=2,
            cells_added_to_hidden=HIDDEN_ROWS * BOARD_COLS,
            cells_with_distribution=0,
            skipped_reason=None,
        )

    if n == 1:
        # 1 セルしか観測 → もう 1 セルが隠し段にある可能性
        # 色は next_pair から残ったほう
        observed_color = new_cells[0][2]
        target_pair = list(prev_next_pair)
        if observed_color in target_pair:
            target_pair.remove(observed_color)
        # 残り色 (target_pair[0]) が隠し段にあるはず
        missing_color = target_pair[0] if target_pair else COLOR_UNKNOWN

        # 落下列の候補: 12 段目 (= row HIDDEN_ROWS) に今回新規出現した列、または
        # 隣接列 (横置きペア)
        observed_row, observed_col, _ = new_cells[0]
        # 候補列: 観測列・左右の列 (ぷよペアは縦置き or 横置き)
        candidate_cols: list[int] = []
        if observed_row == HIDDEN_ROWS:
            # 12 段目 (画面最上段) に積まれた → 観測列が縦置きの 1 個目で、
            # 13 段目に同列の 2 個目がある可能性が高い
            candidate_cols = [observed_col]
        else:
            # 12 段目より下の場合は通常の落下、横置きの可能性
            for dc in (-1, 0, 1):
                cc = observed_col + dc
                if 0 <= cc < BOARD_COLS:
                    candidate_cols.append(cc)

        n_candidates = len(candidate_cols)
        if n_candidates == 0:
            return pboard, HiddenInferenceResult(
                n_new_cells=1, cells_added_to_hidden=0,
                cells_with_distribution=0,
                skipped_reason="no_candidate_columns",
            )

        # 隠し段は基本 EMPTY 確定だが、candidate_cols のみ確率分布
        for c in range(BOARD_COLS):
            if c not in candidate_cols:
                pboard.set_certain(0, c, COLOR_EMPTY)
        prob_per_candidate = 1.0 / n_candidates
        for c in candidate_cols:
            pboard.set_distribution(0, c, {
                missing_color: prob_per_candidate,
                COLOR_EMPTY: 1.0 - prob_per_candidate,
            })

        return pboard, HiddenInferenceResult(
            n_new_cells=1,
            cells_added_to_hidden=BOARD_COLS - n_candidates,
            cells_with_distribution=n_candidates,
            skipped_reason=None,
        )

    # n == 0 or n >= 3: 推論しない (連鎖中・落下中の中途半端)
    return pboard, HiddenInferenceResult(
        n_new_cells=n, cells_added_to_hidden=0,
        cells_with_distribution=0,
        skipped_reason=f"n_new_cells={n}",
    )


# ============================
# Phase I: Platt scaling calibration
# ============================

# Calibration JSON cache (path → params)
# load_calibration() で参照される。fine-tune 後はファイルを書き換えて
# 次の呼び出しで自動再読込されるよう、毎回 mtime を確認する。
_CALIBRATION_CACHE: dict[str, dict[str, float]] = {}
_CALIBRATION_MTIME: dict[str, float] = {}


def _sigmoid(x: float) -> float:
    """1 / (1 + exp(-x)) (overflow ガード付き)."""
    if x < -50.0:
        return 0.0
    if x > 50.0:
        return 1.0
    import math
    return 1.0 / (1.0 + math.exp(-x))


def _load_calibration(
    path: Path | str | None,
) -> dict[str, float] | None:
    """calibration JSON を読み込む. ファイル不在なら None.

    JSON 形式: {"a": float, "b": float, "n_samples": int, "brier_after": float}
    Platt scaling: P_calibrated = sigmoid(a * P_heuristic + b)
    """
    p = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    key = str(p.resolve()) if p.exists() else str(p)
    if not p.is_file():
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    if (
        key in _CALIBRATION_CACHE
        and _CALIBRATION_MTIME.get(key) == mtime
    ):
        return _CALIBRATION_CACHE[key]
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        a = float(data.get("a", 1.0))
        b = float(data.get("b", 0.0))
    except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError):
        return None
    params = {"a": a, "b": b}
    _CALIBRATION_CACHE[key] = params
    _CALIBRATION_MTIME[key] = mtime
    return params


def _apply_calibration_inplace(
    pboard: ProbabilisticBoard,
    calibration_path: Path | str | None,
) -> None:
    """ProbabilisticBoard の隠し段確率に Platt scaling を適用 (in-place).

    各 row 0 セルの「色付き確率 (= EMPTY 以外の総和)」を p とし、
    p_new = sigmoid(a * p + b) に補正、EMPTY 確率を 1 - p_new に置く。
    色配分は元の比率を保持する。
    """
    params = _load_calibration(calibration_path)
    if params is None:
        return
    a = params["a"]
    b = params["b"]
    for col in range(BOARD_COLS):
        cell = pboard.cell(0, col)
        if not cell.probs:
            continue
        # 色付き確率の合計と内訳
        colored_total = 0.0
        colored: dict[int, float] = {}
        for color, prob in cell.probs.items():
            if color in (COLOR_EMPTY, COLOR_UNKNOWN):
                continue
            colored[color] = float(prob)
            colored_total += float(prob)
        # 色付き確率が 0 または 1 の極端ケースは補正しない
        if colored_total <= 1e-9 or colored_total >= 1.0 - 1e-9:
            continue
        # Platt scaling
        new_colored_total = _sigmoid(a * colored_total + b)
        # 内部比率を保ったまま色付き確率を再分配
        new_probs: dict[int, float] = {}
        scale = new_colored_total / colored_total
        for color, prob in colored.items():
            new_probs[color] = prob * scale
        new_probs[COLOR_EMPTY] = max(0.0, 1.0 - new_colored_total)
        pboard.set_distribution(0, col, new_probs)


__all__ = [
    "DEFAULT_CALIBRATION_PATH",
    "HiddenInferenceResult",
    "SKIP_COLORS",
    "infer_hidden_row",
]
