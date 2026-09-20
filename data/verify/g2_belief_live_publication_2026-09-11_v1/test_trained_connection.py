"""修復binder→学習済みfold6三seed。盤面/反映機構は人工fixtureで品質測定ではない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pytest
import torch
from test_live_binding import saved, live
from test_live_binding_v2 import connected
import trained_connection as T

ROOT = Path(__file__).resolve().parent
SAMPLES, SEED = 32, 17


def test_fixed_trained_members(connected: Any) -> None:
    torch.set_num_threads(2)
    score = T.backend()
    loader = score.load_loader()
    members = loader.load_members(loader.SOURCE_ID, 'cpu')
    bound, result, details = T.evaluate(connected.capture, members, sample_count=SAMPLES, seed=SEED)
    expected = float(np.mean(list(details['seed_calibrated'].values())))
    assert result.probability_p1 == pytest.approx(expected, abs=1e-12)
    assert result.world_counts == (49, 1) and result.used_samples == SAMPLES
    assert len(details['members']) == 3 and not result.quality_gate_clear
    assert not torch.cuda.is_initialized()
    record = dict(frame=bound.frame, probability_p1=result.probability_p1, details=details,
        original_context_run=bound.values[0].scope[1], artificial_pipe_and_second_basis=True,
        actual_video=False, quality_gate_clear=False)
    with (ROOT/'TRAINED_CONNECTION_RESULT_v1.json').open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
