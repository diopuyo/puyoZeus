"""ScoreFineTuner: score OCR digit テンプレートを擬似ラベルで update する.

実装方針:
    score_ocr.py は NCC テンプレマッチで digit_0..digit_9 を読み取る。
    本 fine-tuner は擬似ラベルから digit_X.png を 平均化更新 して保存する。

    1. 擬似ラベルから (digit_label, patch) を集約
    2. label 別に patch を中央値合成 (or 平均) → 新テンプレ
    3. 既存テンプレと alpha-blend して滑らかに更新 (lr=0.3)
    4. テンプレ保存 (`models/ui_templates/score_digits/digit_N.png`)
    5. rollback 用に backup を取る
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.score_ocr import (
    DEFAULT_TEMPLATE_DIR,
    DIGIT_HEIGHT,
    DIGIT_LABELS,
    DIGIT_WIDTH,
)
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.pseudo_label import (
    COMPONENT_SCORE,
    PseudoLabelSample,
)


# 既存テンプレと擬似ラベル平均の混合率
DEFAULT_LR: float = 0.30
# 1 label あたり最低必要サンプル数 (これ未満は更新しない)
MIN_SAMPLES_PER_LABEL: int = 5


@dataclass
class ScoreFineTuneMetrics:
    """fine_tune の戻り値."""

    n_samples: int
    n_labels_updated: int
    samples_per_label: dict[int, int]


class ScoreFineTuner(OnlineFineTuner):
    """ScoreOcr の digit テンプレを擬似ラベルで更新する fine-tuner."""

    def __init__(
        self,
        template_dir: Path | str = DEFAULT_TEMPLATE_DIR,
        backup_suffix: str = ".bak",
        lr: float = DEFAULT_LR,
    ) -> None:
        self._template_dir = Path(template_dir)
        self._template_dir.mkdir(parents=True, exist_ok=True)
        self._backup_suffix = str(backup_suffix)
        if not (0.0 < lr <= 1.0):
            raise ValueError("lr must be in (0, 1]")
        self._lr = float(lr)
        self._backup_done: bool = False

    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベルから per-digit テンプレを更新."""
        # filter: component=score かつ patch 入り
        useful = [
            s for s in samples
            if s.component == COMPONENT_SCORE
            and isinstance(s.input_data, dict)
            and "patch" in s.input_data
            and isinstance(s.label, int)
        ]
        if not useful:
            return ScoreFineTuneMetrics(
                n_samples=0, n_labels_updated=0,
                samples_per_label={},
            ).__dict__
        # label 別グループ化
        groups: dict[int, list[np.ndarray]] = {}
        for s in useful:
            patch = s.input_data["patch"]
            if not isinstance(patch, np.ndarray):
                continue
            patch_resized = self._normalize_patch(patch)
            if patch_resized is None:
                continue
            label = int(s.label)
            if label < 0 or label > 9:
                continue
            groups.setdefault(label, []).append(patch_resized)
        # backup
        if not self._backup_done:
            self._make_backup()
            self._backup_done = True
        # 各 label でテンプレ更新
        n_updated = 0
        per_label: dict[int, int] = {}
        for label in DIGIT_LABELS:
            patches = groups.get(label, [])
            per_label[label] = len(patches)
            if len(patches) < MIN_SAMPLES_PER_LABEL:
                continue
            stacked = np.stack(patches, axis=0)
            new_tpl = np.median(stacked, axis=0).astype(np.uint8)
            blended = self._blend(label, new_tpl)
            self._save_template(label, blended)
            n_updated += 1
        return ScoreFineTuneMetrics(
            n_samples=len(useful),
            n_labels_updated=n_updated,
            samples_per_label=per_label,
        ).__dict__

    def rollback(self) -> None:
        """backup から復元."""
        for label in DIGIT_LABELS:
            target = self._template_path(label)
            backup = target.with_suffix(target.suffix + self._backup_suffix)
            if backup.is_file():
                shutil.copyfile(backup, target)
        self._backup_done = False

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _template_path(self, label: int) -> Path:
        return self._template_dir / f"digit_{label}.png"

    def _make_backup(self) -> None:
        for label in DIGIT_LABELS:
            tpl = self._template_path(label)
            if tpl.is_file():
                backup = tpl.with_suffix(tpl.suffix + self._backup_suffix)
                shutil.copyfile(tpl, backup)

    def _save_template(self, label: int, img: np.ndarray) -> None:
        """テンプレ画像を保存 (BGR or grayscale)."""
        path = self._template_path(label)
        if img.ndim == 2:
            cv2.imwrite(str(path), img)
        else:
            cv2.imwrite(str(path), img)

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray | None:
        """patch サイズを (DIGIT_HEIGHT, DIGIT_WIDTH, 3) に揃える."""
        if patch is None or patch.size == 0:
            return None
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        if patch.ndim != 3:
            return None
        h, w = patch.shape[:2]
        if (h, w) != (DIGIT_HEIGHT, DIGIT_WIDTH):
            patch = cv2.resize(
                patch, (DIGIT_WIDTH, DIGIT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        return patch

    def _blend(self, label: int, new_tpl: np.ndarray) -> np.ndarray:
        """既存テンプレと lr-blend (移動平均的)."""
        old_path = self._template_path(label)
        if not old_path.is_file():
            return new_tpl
        old = cv2.imread(str(old_path))
        if old is None:
            return new_tpl
        if old.shape != new_tpl.shape:
            old = cv2.resize(
                old, (new_tpl.shape[1], new_tpl.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        blended = (
            (1.0 - self._lr) * old.astype(np.float32)
            + self._lr * new_tpl.astype(np.float32)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)


__all__ = [
    "DEFAULT_LR",
    "MIN_SAMPLES_PER_LABEL",
    "ScoreFineTuneMetrics",
    "ScoreFineTuner",
]
