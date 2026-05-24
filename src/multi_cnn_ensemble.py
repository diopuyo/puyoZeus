"""Multi-CNN ensemble (Z-X 候補): v16/v17/v17b の predict_proba を平均化。

各動画別に強み弱みがあるため、複数 CNN の確率分布を平均化して
ロバストな予測を狙う。動画別 model 選択は別大会動画で機能しないので、
すべての CNN を実行して結果を統合する方式。

設計:
    - 複数 CnnPatchClassifier を保持
    - predict_proba_batch で各 model の確率取得
    - 重み付き平均 (default 等重み)
    - argmax で最終 color

統合:
    - HybridClassifier に近い構造、cnn_classifier 引数として使用可
    - StatePipeline は既存のまま使える (HybridClassifier を ensemble で wrap)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.patch_classifier import CLASS_INDEX_TO_COLOR, NUM_CLASSES


@dataclass
class MultiCnnEnsemble:
    """複数 CNN の predict_proba を平均化する分類器。

    HybridClassifier の cnn_classifier として使用可能 (predict_proba と
    predict_proba_batch 互換 API)。
    """
    cnns: list  # list[CnnPatchClassifier]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            n = len(self.cnns)
            self.weights = [1.0 / n] * n if n > 0 else []
        if len(self.weights) != len(self.cnns):
            raise ValueError(
                f"weights {len(self.weights)} != cnns {len(self.cnns)}"
            )
        # 正規化
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

    def predict_proba(self, bgr_patch: np.ndarray) -> np.ndarray:
        """各 CNN の確率を重み付き平均。"""
        if not self.cnns:
            return np.zeros(NUM_CLASSES, dtype=np.float32)
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        for cnn, w in zip(self.cnns, self.weights):
            try:
                p = cnn.predict_proba(bgr_patch)
                probs += w * p
            except Exception:
                pass
        return probs

    def predict_proba_batch(
        self, bgr_patches: list[np.ndarray],
    ) -> np.ndarray:
        """バッチ predict_proba を重み付き平均。"""
        if not bgr_patches or not self.cnns:
            return np.zeros((len(bgr_patches), NUM_CLASSES), dtype=np.float32)
        # 各 CNN のバッチ確率を集計
        n = len(bgr_patches)
        ensemble_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        for cnn, w in zip(self.cnns, self.weights):
            if hasattr(cnn, "predict_proba_batch"):
                try:
                    probs = cnn.predict_proba_batch(bgr_patches)
                    ensemble_probs += w * probs
                except Exception:
                    pass
            else:
                # fallback: 個別呼び出し
                for i, p in enumerate(bgr_patches):
                    try:
                        ensemble_probs[i] += w * cnn.predict_proba(p)
                    except Exception:
                        pass
        return ensemble_probs

    def classify(self, bgr_patch: np.ndarray) -> int:
        probs = self.predict_proba(bgr_patch)
        idx = int(np.argmax(probs))
        return CLASS_INDEX_TO_COLOR[idx]

    def classify_batch(self, bgr_patches: list[np.ndarray]) -> list[int]:
        probs = self.predict_proba_batch(bgr_patches)
        idxs = np.argmax(probs, axis=1)
        return [CLASS_INDEX_TO_COLOR[int(i)] for i in idxs]


def load_ensemble_v16_v17b(
    device: str = "cpu",
) -> MultiCnnEnsemble:
    """v16 + v17b の 2 model ensemble をロード (デフォルト等重み)。"""
    import torch
    from src.patch_classifier import CnnPatchClassifier
    cnns = []
    for path in (
        "models/cnn_phase_u_v16.pt",
        "models/cnn_phase_u_v17b.pt",
    ):
        cnn = CnnPatchClassifier()
        state = torch.load(path, map_location="cpu", weights_only=True)
        cnn._model.load_state_dict(state)
        cnn._model.eval()
        if hasattr(cnn, "to_device"):
            try:
                cnn.to_device(device)
            except Exception:
                pass
        cnns.append(cnn)
    return MultiCnnEnsemble(cnns=cnns)


__all__ = [
    "MultiCnnEnsemble",
    "load_ensemble_v16_v17b",
]
