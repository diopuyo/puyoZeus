"""NextFineTuner: next/dnext 検出 CNN を擬似ラベルで fine-tune.

実装方針:
    既存 NextDetector は CnnPatchClassifier を内部で利用している。
    本 fine-tuner は擬似ラベルから (patch, color_label) を集約し、
    既存 model に対して AdamW (lr=1e-4) で 1〜数 epoch fine-tune。

注意:
    NextDetector の内部 CNN は GatedCnnClassifier 経由なので、
    fine-tune 対象は GatedCnnClassifier._color (= CnnPatchClassifier) または
    与えられた CnnPatchClassifier 直接。

    fine-tune 後の state を rollback できるよう、初期 state_dict を保存。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.patch_classifier import CnnPatchClassifier
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.pseudo_label import (
    COMPONENT_NEXT,
    PseudoLabelSample,
)


# fine-tune ハイパラ
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 32
MIN_TOTAL_SAMPLES: int = 20  # これ未満なら fine-tune skip


@dataclass
class NextFineTuneMetrics:
    """fine_tune の戻り値."""

    n_samples: int
    n_epochs: int
    loss_before: float
    loss_after: float
    samples_per_label: dict[int, int] = field(default_factory=dict)


class NextFineTuner(OnlineFineTuner):
    """NextDetector 内部 CNN を擬似ラベルで fine-tune."""

    def __init__(
        self,
        cnn: CnnPatchClassifier,
        lr: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if cnn is None:
            raise ValueError("cnn is required")
        if lr <= 0 or epochs <= 0 or batch_size <= 0:
            raise ValueError("lr/epochs/batch_size must be positive")
        self._cnn = cnn
        self._lr = float(lr)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._backup_state: Any | None = None

    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベルから (patch, color_label) を抽出して 1 epoch fine-tune.

        擬似ラベルは {top, bot} の patch + label を持つので、両方を学習に使う。
        """
        # filter
        x_list, y_list = self._collect_pairs(samples)
        if len(x_list) < MIN_TOTAL_SAMPLES:
            return NextFineTuneMetrics(
                n_samples=len(x_list), n_epochs=0,
                loss_before=0.0, loss_after=0.0,
                samples_per_label=self._count_per_label(y_list),
            ).__dict__
        if self._backup_state is None:
            self._save_backup()
        loss_before = self._compute_loss(x_list, y_list)
        self._train_loop(x_list, y_list)
        loss_after = self._compute_loss(x_list, y_list)
        return NextFineTuneMetrics(
            n_samples=len(x_list),
            n_epochs=self._epochs,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            samples_per_label=self._count_per_label(y_list),
        ).__dict__

    def rollback(self) -> None:
        """fine-tune 前の state_dict に巻き戻し."""
        if self._backup_state is None:
            return
        self._cnn._model.load_state_dict(self._backup_state)
        self._backup_state = None

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _collect_pairs(
        self, samples: list[PseudoLabelSample],
    ) -> tuple[list[np.ndarray], list[int]]:
        """擬似ラベルから (patch, color_label) を抽出."""
        from src.patch_classifier import COLOR_TO_CLASS_INDEX
        color_to_idx: dict[int, int] = dict(COLOR_TO_CLASS_INDEX)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for s in samples:
            if s.component != COMPONENT_NEXT:
                continue
            if not isinstance(s.input_data, dict):
                continue
            if not isinstance(s.label, dict):
                continue
            top_color = s.label.get("top_color")
            bot_color = s.label.get("bot_color")
            patch_top = s.input_data.get("patch_top")
            patch_bot = s.input_data.get("patch_bot")
            for patch, color in (
                (patch_top, top_color), (patch_bot, bot_color),
            ):
                if patch is None or color is None:
                    continue
                if not isinstance(patch, np.ndarray):
                    continue
                idx = color_to_idx.get(int(color))
                if idx is None:
                    continue
                xs.append(patch)
                ys.append(int(idx))
        return xs, ys

    @staticmethod
    def _count_per_label(ys: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for y in ys:
            out[y] = out.get(y, 0) + 1
        return out

    def _save_backup(self) -> None:
        """現 state_dict を backup."""
        torch = self._cnn._torch
        self._backup_state = {
            k: v.detach().clone()
            for k, v in self._cnn._model.state_dict().items()
        }

    def _compute_loss(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> float:
        """全サンプルでの cross-entropy loss を計算 (fine-tune 前後の比較用)."""
        torch = self._cnn._torch
        if not xs:
            return 0.0
        self._cnn._model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(xs), self._batch_size):
                bx = xs[i:i + self._batch_size]
                by = ys[i:i + self._batch_size]
                tensors = [self._cnn._patch_to_tensor(p)[0] for p in bx]
                batch = torch.stack(tensors).to(self._cnn._device)
                target = torch.tensor(by, dtype=torch.long).to(
                    self._cnn._device,
                )
                logits = self._cnn._model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits, target, reduction="sum",
                )
                total += float(loss.item())
                n += len(bx)
        return total / max(1, n)

    def _train_loop(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> None:
        """AdamW で N epoch 学習."""
        torch = self._cnn._torch
        self._cnn._model.train()
        opt = torch.optim.AdamW(
            self._cnn._model.parameters(), lr=self._lr,
        )
        rng = np.random.default_rng(seed=0)
        for _ in range(self._epochs):
            idx = rng.permutation(len(xs))
            for i in range(0, len(xs), self._batch_size):
                batch_idx = idx[i:i + self._batch_size]
                bx = [xs[j] for j in batch_idx]
                by = [ys[j] for j in batch_idx]
                tensors = [self._cnn._patch_to_tensor(p)[0] for p in bx]
                batch = torch.stack(tensors).to(self._cnn._device)
                target = torch.tensor(by, dtype=torch.long).to(
                    self._cnn._device,
                )
                logits = self._cnn._model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits, target,
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._cnn._model.eval()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "MIN_TOTAL_SAMPLES",
    "NextFineTuneMetrics",
    "NextFineTuner",
]
