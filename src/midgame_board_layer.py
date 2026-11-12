"""中盤の盤面補正。盤の判定と計算は状態を持たない。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

ENV_ENABLED = 'PUYO_MIDGAME_BOARD_LAYER'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/calibration/midgame_board_layer.json'
MODEL_SHA256 = 'd396edb5dbf28d16dfebc3eaaa8cb67f1c02cc1721fe0417eb28526de2d45834'
ROWS, COLS = 13, 6
COLORS = frozenset(range(1, 6))
VALID = COLORS | {0, 9}
ERASE_COUNT = 4
EARLY_MAX, LATE_MIN, COLOR_DIFF_MAX = 18, 48, 4
PROB_CLIP = 1e-6
ZERO_FIELDS = ('chain_active', 'sl_pending', 'r_ledger_forecast', 'cells_ojama')


def load_layer(path: Path = MODEL_PATH, enabled: bool | None = None) -> dict | None:
    """不一致・欠損は記録して無効化する。OFFではファイルも読まない。"""
    if enabled is None:
        from src.production_config import DISTRIBUTION_MIDGAME_BOARD_LAYER
        enabled = os.environ.get(ENV_ENABLED, str(int(DISTRIBUTION_MIDGAME_BOARD_LAYER))) == '1'
    if not enabled:
        return None
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError('SHA256不一致')
        model = json.loads(raw)
        for name in ('sd_bump', 'sd_color'):
            if not math.isfinite(model['shift'][name]) or model['shift'][name] <= 0:
                raise ValueError(name)
        for name in ('b_bump', 'b_color'):
            if not math.isfinite(model['shift'][name]):
                raise ValueError(name)
        for name in ('cap_logit', 'enter_sec', 'exit_sec'):
            if not math.isfinite(model[name]) or model[name] <= 0:
                raise ValueError(name)
        return model
    except (OSError, ValueError, KeyError, TypeError) as error:
        logging.getLogger(__name__).warning('中盤補正を無効化: %s: %s', path, error)
        return None


def settled_features(board: Sequence[Sequence[int]] | None) -> tuple[int, int] | None:
    """全13段の色数と凹凸を返す。不明・浮き・消える群は採用しない。"""
    if board is None or len(board) != ROWS or any(len(row) != COLS for row in board):
        return None
    if any(cell not in VALID for row in board for cell in row):
        return None
    heights = [sum(board[r][c] > 0 for r in range(ROWS)) for c in range(COLS)]
    if any(board[r][c] > 0 and board[r + 1][c] == 0
           for r in range(ROWS - 1) for c in range(COLS)):
        return None
    if has_erasing_group(board):
        return None
    color = sum(cell in COLORS for row in board for cell in row)
    bump = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    return int(color), int(bump)


def has_erasing_group(board: Sequence[Sequence[int]]) -> bool:
    """色1〜5の上下左右の連結を調べる。"""
    seen = set()
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] not in COLORS or (r, c) in seen:
                continue
            stack, count = [(r, c)], 0
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                count += 1
                if count >= ERASE_COUNT:
                    return True
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < ROWS and 0 <= nx < COLS and (ny, nx) not in seen
                            and board[ny][nx] == board[r][c]):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return False


def eligible(row: Mapping[str, Any], counter_window: bool = False) -> bool:
    """対象条件は現在行、特徴量は持ち越した確定盤から得る。欠測は対象外。"""
    if counter_window:
        return False
    try:
        if any(float(row[f'{name}_p{s}']) != 0 for name in ZERO_FIELDS for s in (1, 2)):
            return False
        totals = [float(row[f'cells_total_p{s}']) for s in (1, 2)]
        colors = [float(row[f'cells_color_p{s}']) for s in (1, 2)]
        if not all(math.isfinite(v) and v >= 0 for v in totals + colors):
            return False
        return EARLY_MAX < max(totals) < LATE_MIN and abs(colors[0] - colors[1]) <= COLOR_DIFF_MAX
    except (KeyError, TypeError, ValueError):
        return False


def board_shift(first: tuple[int, int], second: tuple[int, int], model: Mapping) -> float:
    """研究仕様の差分を各標準偏差で正規化し、JSONの上限内に収める。"""
    coef = model['shift']
    value = (coef['b_color'] * (first[0] - second[0]) / coef['sd_color']
             + coef['b_bump'] * (first[1] - second[1]) / coef['sd_bump'])
    return max(-model['cap_logit'], min(model['cap_logit'], value))


def shift_probability(probability: float, shift: float) -> float:
    """補正0ではclipもせず、元の値をそのまま返す。"""
    if shift == 0:
        return probability
    p = min(1 - PROB_CLIP, max(PROB_CLIP, probability))
    return 1 / (1 + math.exp(-(math.log(p / (1 - p)) + shift)))
