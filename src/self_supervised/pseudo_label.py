"""擬似ラベルのデータクラス定義.

CrossValidator が抽出する 1 件のサンプル。
component / timestamp / input_data / label / confidence / metadata を保持。
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# サポートする component (拡張時はここに追加)
COMPONENT_SCORE: str = "score"
COMPONENT_NEXT: str = "next"
COMPONENT_CHAIN: str = "chain"
COMPONENT_CELL: str = "cell"  # cell-level (= color 分類用) pseudo label
COMPONENT_FRAME_STATE: str = "frame_state"  # frame-level state ラベル
# (= STABLE/CHAIN/EFFECT/OJAMA_FALL/MENU、 X-2 提案、 2026-05-12)

# 信頼度の最低閾値 (この値未満は破棄推奨)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class PseudoLabelSample:
    """擬似ラベル 1 件.

    Attributes:
        component: "score" / "next" / "chain" / "cell"
        timestamp: フレーム時刻 (秒)。同一 video 内 unique 想定。
        input_data: 入力データ。numpy 画像 / dict / tuple 等を想定。
            JSONL に乗せる際は serialize_input() を介して変換する。
        label: 正解ラベル。スカラ / tuple / dict 等。
        confidence: 自己整合性の一致度 [0, 1]。
        metadata: debug 用辞書 (frame_idx, video_id, 比較材料 等)。
    """

    component: str
    timestamp: float
    input_data: Any
    label: Any
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        """JSONL 行に変換可能な dict を返す."""
        return {
            "component": self.component,
            "timestamp": float(self.timestamp),
            "input_data": _to_jsonable(self.input_data),
            "label": _to_jsonable(self.label),
            "confidence": float(self.confidence),
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_jsonable(cls, obj: dict[str, Any]) -> "PseudoLabelSample":
        """JSONL 行 dict から復元."""
        return cls(
            component=str(obj["component"]),
            timestamp=float(obj["timestamp"]),
            input_data=_from_jsonable(obj.get("input_data")),
            label=_from_jsonable(obj.get("label")),
            confidence=float(obj.get("confidence", 0.0)),
            metadata=dict(obj.get("metadata", {}) or {}),
        )


def _to_jsonable(obj: Any) -> Any:
    """numpy 配列 / tuple 等を JSON 化可能な形へ.

    画像 (uint8 ndarray) は base64 で encode する。
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # numpy scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _from_jsonable(obj: Any) -> Any:
    """_to_jsonable の逆変換."""
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        return _decode_ndarray(obj)
    if isinstance(obj, list):
        return [_from_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _from_jsonable(v) for k, v in obj.items()}
    return obj


def _encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    """numpy 配列を JSON 化可能な dict に変換 (base64)."""
    return {
        "__ndarray__": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_ndarray(d: dict[str, Any]) -> np.ndarray:
    """_encode_ndarray の逆変換."""
    raw = base64.b64decode(d["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"]))
    return arr.reshape(tuple(d["shape"]))


__all__ = [
    "COMPONENT_CELL",
    "COMPONENT_CHAIN",
    "COMPONENT_FRAME_STATE",
    "COMPONENT_NEXT",
    "COMPONENT_SCORE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "PseudoLabelSample",
]
