"""OnlineFineTuner: 擬似ラベルから model fine-tune を行う基底クラス.

各 component (Score / Next / Cell) で個別の subclass を提供する。
fine-tune は GPU 上で実行可能。rollback() で直前状態に巻き戻せる。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.self_supervised.pseudo_label import PseudoLabelSample


class OnlineFineTuner(ABC):
    """擬似ラベル fine-tune 基底.

    Subclass の責務:
        - fine_tune(): 受け取った擬似ラベルで model 更新
                       戻り値は metrics dict {acc_before, acc_after, n_samples, ...}
        - rollback(): fine_tune 前の state に巻き戻し
    """

    @abstractmethod
    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベルで model を fine-tune.

        Returns:
            metrics dict、最低限 {n_samples: int} を含む
        """
        raise NotImplementedError

    @abstractmethod
    def rollback(self) -> None:
        """直前 fine_tune を巻き戻し (backup state を復元)."""
        raise NotImplementedError


__all__ = ["OnlineFineTuner"]
