"""最終学習済みM0をG_feの確定盤面入力へ接続する。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV2, board_categories_from_raw,
)
from src.exchange_event_evaluator import file_sha256

PROBABILITY_EPSILON = 1e-7
LOGIT_LIMIT = 30.0
CPU_THREADS = 2


class FileM0Predictor:
    """推論はCPUで実行し、認識CNNとGPUメモリを競合させない。"""

    def __init__(self, directory: Path) -> None:
        torch.set_num_threads(CPU_THREADS)
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = directory / "model.pt"
        if file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("M0成果物のハッシュが不一致")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = AdvantageM0CurrentCNNV2()
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.slope = float(saved["slope"])

    def __call__(self, boards: np.ndarray, queues: np.ndarray) -> float:
        """T03と同じ盤面category化・queue欠測処理・対称較正を適用する。"""
        categories = np.stack([board_categories_from_raw(b) for b in boards])
        queue = np.where((queues >= 1) & (queues <= 5), queues, 0)
        with torch.inference_mode():
            output = self.model(torch.as_tensor(categories[None], dtype=torch.long),
                                torch.as_tensor(queue[None], dtype=torch.long))
        probability = np.clip(float(output.raw_probability.item()),
                              PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        logit = np.log(probability / (1 - probability)) * self.slope
        return float(1 / (1 + np.exp(-np.clip(logit, -LOGIT_LIMIT, LOGIT_LIMIT))))


def formula_totals_from_pipeline(pipeline: object) -> tuple[float | None, float | None]:
    """既存段累積器の実観測だけを読み取り、推定chain.total_scoreで補わない。"""
    values = []
    for label in ("1p", "2p"):
        accumulator = getattr(pipeline, "_formula_accum_" + label, None)
        values.append(accumulator.total_power if accumulator is not None
                      and accumulator.step_count else None)
    return tuple(values)
