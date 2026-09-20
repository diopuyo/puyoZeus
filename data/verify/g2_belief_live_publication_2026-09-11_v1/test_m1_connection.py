"""生Registry接続fixture→実M1 API。重みは人工で、モデル品質評価ではない。"""
from __future__ import annotations

from typing import Any
import numpy as np
import pytest
import torch
from test_live_binding import saved, live
import m1_connection as M
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2, board_categories_from_raw
from src.advantage_m1_zero_counterfactual_v3 import AdvantageM1ZeroCounterfactualV3

SAMPLES, SEED = 32, 17


def model() -> Any:
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    return AdvantageM1ZeroCounterfactualV3(AdvantageM0CurrentCNNV2(), 'values_and_masks').eval()


def test_actual_m1_unknown_ledger_branch(live: Any) -> None:
    network = model()
    bound, result = M.evaluate(live.capture, network, sample_count=SAMPLES, seed=SEED)
    samples = M.L.J.draw(bound.values, SAMPLES, SEED)
    boards = np.stack([np.stack([board_categories_from_raw(side) for side in pair]) for pair in samples])
    queues = np.repeat(bound.inputs.queues[None], SAMPLES, axis=0)
    with torch.no_grad():
        direct = network.m0(torch.tensor(boards, dtype=torch.int64), torch.tensor(queues, dtype=torch.int64)).raw_probability
    assert result.probability_p1 == pytest.approx(float(direct.mean()), abs=1e-7)
    assert result.world_counts == (49, 1) and result.used_samples == SAMPLES
    assert not result.accounting_permission and not result.quality_gate_clear
    assert all(parameter.grad is None for parameter in network.parameters())


def test_training_model_rejected(live: Any) -> None:
    with pytest.raises(ValueError, match='model_must_be_eval'):
        M.evaluate(live.capture, model().train(), sample_count=SAMPLES)
