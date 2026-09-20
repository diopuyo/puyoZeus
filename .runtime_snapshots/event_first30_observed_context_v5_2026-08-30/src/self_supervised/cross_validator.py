"""CrossValidator: 自己整合性検査による擬似ラベル抽出 base.

各コンポーネント (Score / Next / ChainEvent) は本クラスを継承し、
update() で pipeline の 1 frame 結果を受け取り、内部に状態を蓄積。
collect() で蓄積された擬似ラベルを返す。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.self_supervised.pseudo_label import PseudoLabelSample


class CrossValidator(ABC):
    """擬似ラベル抽出の抽象基底.

    Subclass の責務:
        - update(): 1 frame の pipeline 結果と raw frame を受け取り、
          内部 state を更新。確定した擬似ラベルがあれば内部 buffer に蓄積。
        - collect(): buffer 内容を list[PseudoLabelSample] で返し、buffer を空に。

    状態保持の利点:
        - 「N frame 連続同一 → 確定」のような時間的整合性を見られる
        - 「次の STABLE で配置色 delta 一致 → 過去の next が正解」など
          後付け確定パターンを実装できる
    """

    def __init__(self) -> None:
        self._buffer: list[PseudoLabelSample] = []

    @abstractmethod
    def update(
        self,
        frame_idx: int,
        t_sec: float,
        pipeline_result: Any,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """1 frame の更新.

        Args:
            frame_idx: フレーム番号 (連番)
            t_sec: フレーム時刻 (秒)
            pipeline_result: PipelineResult (循環 import 回避のため Any)
            frame_bgr: 元 BGR フレーム (擬似ラベルの input_data に使う場合あり)
        """
        raise NotImplementedError

    def collect(self) -> list[PseudoLabelSample]:
        """蓄積された擬似ラベルを取り出して buffer を空にする."""
        out = list(self._buffer)
        self._buffer.clear()
        return out

    def reset(self) -> None:
        """state + buffer を完全クリア (試合切替時など)."""
        self._buffer.clear()

    def _emit(self, sample: PseudoLabelSample) -> None:
        """サブクラス用 helper: buffer に擬似ラベルを 1 件追加."""
        self._buffer.append(sample)


__all__ = ["CrossValidator"]
