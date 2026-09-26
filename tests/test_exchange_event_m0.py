"""M0成果物のロード、学習時との前処理一致、左右対称性を確認する。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.exchange_event_evaluator import file_sha256
from src.exchange_event_m0 import FileM0Predictor


@pytest.fixture
def predictor(tmp_path: Path) -> FileM0Predictor:
    """小さな実モデルを保存し、実際のロード経路を通す。"""
    model = AdvantageM0CurrentCNNV2()
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(state_dict=model.state_dict(), slope=.7), checkpoint)
    (tmp_path / "manifest.json").write_text(json.dumps(
        dict(checkpoint_sha256=file_sha256(checkpoint))))
    return FileM0Predictor(tmp_path)


def test_raw_preprocessing_and_calibration_match_training(predictor: FileM0Predictor) -> None:
    from scripts.train_advantage_m1_causal_ledger_v1 import calibrate
    raw = np.zeros((2, 13, 6), dtype=np.int8)
    raw[0, -1] = [1, 2, 3, 4, 9, 10]
    queue = np.array([[1, 2, 9, 10], [3, 4, 0, 5]])
    categories = raw.copy()
    categories[raw == 9], categories[raw == 10] = 6, 7
    clean_queue = np.where((queue >= 1) & (queue <= 5), queue, 0)
    with torch.inference_mode():
        expected = predictor.model(torch.tensor(categories[None], dtype=torch.long),
                                   torch.tensor(clean_queue[None], dtype=torch.long))
    assert predictor(raw, queue) == pytest.approx(
        calibrate(expected.raw_probability.numpy(), predictor.slope)[0])
    assert predictor(raw[::-1].copy(), queue[::-1].copy()) == pytest.approx(
        1 - predictor(raw, queue), abs=1e-6)


def test_m0_hash_mismatch_is_rejected(predictor: FileM0Predictor, tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="ハッシュ"):
        FileM0Predictor(tmp_path)
