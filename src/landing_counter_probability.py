"""受け側視点の観測特徴と、差し替え可能な応手確率モデル。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
import numpy as np
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, DEATH_COL

MODEL_PATH = Path("models/landing_counter_prob_v1/model.json")
COLUMNS = ("send_ratio", "hands", "max_height", "death_column_height", "occupancy",
           "ojama_fraction", "queue_known", "board_age", "already_chaining")


def bounded(value: float) -> float:
    """非負量を順序を保ったまま0〜1へ写す。"""
    value = max(0., value)
    return value/(1.+value)


def response_features(grid: np.ndarray, queue: np.ndarray, send: float, incoming: float,
                      hands: int, board_age: float, chaining: bool) -> np.ndarray:
    """現在得られている受け側の量だけを使い、左右で符号反転しない。"""
    occupied = grid != 0
    heights = [BOARD_ROWS-int(np.flatnonzero(occupied[:, c])[0]) if occupied[:, c].any() else 0
               for c in range(BOARD_COLS)]
    return np.array([bounded(send/max(incoming, 1.)), bounded(hands),
        max(heights)/BOARD_ROWS, heights[DEATH_COL]/BOARD_ROWS, float(occupied.mean()),
        float((grid == COLOR_OJAMA).mean()), float(((queue >= 1) & (queue <= 5)).mean()),
        bounded(board_age), float(chaining)], dtype=np.float64)


class ResponseProbability(Protocol):
    """学習器の形式を仮想着弾から切り離す。"""
    def predict(self, features: np.ndarray) -> float: ...


@dataclass(frozen=True)
class LogisticResponseProbability:
    """動画分割で学習した標準化＋ロジスティック回帰を軽量JSONから読む。"""
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> LogisticResponseProbability:
        """特徴順の不一致や、未完了モデルを黙って採用しない。"""
        value = json.loads(path.read_text(encoding="utf-8"))
        if tuple(value["columns"]) != COLUMNS or not value["valid"]:
            raise ValueError("応手確率モデルの特徴順または完了状態が不正")
        arrays = [np.asarray(value[k], dtype=float) for k in ("mean", "scale", "coef")]
        if any(a.shape != (len(COLUMNS),) or not np.isfinite(a).all() for a in arrays) or (arrays[1] <= 0).any():
            raise ValueError("応手確率モデルの係数が不正")
        return cls(*arrays, float(value["intercept"]))

    def predict(self, features: np.ndarray) -> float:
        """正規化は学習fold由来の値に固定し、将来情報を使用しない。"""
        z = float(np.dot((features-self.mean)/self.scale, self.coef)+self.intercept)
        return float(np.exp(-np.logaddexp(0., -z)))
