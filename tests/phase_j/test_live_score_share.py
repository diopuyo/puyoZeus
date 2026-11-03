"""同一フレーム・同一OCR条件だけを再利用する契約。"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.phase_j.live_score_share import RawScores, compatible, prepare, shared_scores
from src.score_ocr import ScoreOcr


def ocr(**kwargs: Any) -> ScoreOcr:
    return ScoreOcr({0: np.full((50, 40), 200, dtype=np.uint8)}, **kwargs)


@pytest.mark.parametrize('kwargs', [dict(min_confidence=0.12), dict(margin_min=0.13),
    dict(avg_min_confidence=0.14), dict(enable_matmul_ncc=True), dict(region_offsets={'1P': (1, 0)})])
def test_conditions_must_match(kwargs: dict) -> None:
    assert not compatible(ocr(), ocr(**kwargs))


def test_template_and_type_must_match() -> None:
    left, right = ocr(), ocr()
    assert compatible(left, right)
    right._templates_gray[0][0, 0] = 10
    assert not compatible(left, right)
    assert not compatible(None, left)


def test_missing_score_is_not_previous_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_B3_SHARE_SCORE_READ', '1')
    owner = SimpleNamespace(_score_ocr=ocr())
    frame = SimpleNamespace(index=2, media_sec=0.5)
    prepare(owner)
    assert shared_scores(owner, ocr(), frame) is None
    owner._live_raw_scores = (2, 0.5, None, 120)
    assert shared_scores(owner, ocr(), frame) == RawScores(None, 120)
    assert shared_scores(owner, ocr(), SimpleNamespace(index=3, media_sec=0.6)) is None
    prepare(owner)
    assert shared_scores(owner, ocr(), frame) is None


def test_off_and_launcher_delivery(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    from src.phase_j.live_speed import validate
    monkeypatch.delenv('PUYO_B3_SHARE_SCORE_READ', raising=False)
    owner = SimpleNamespace(_live_raw_scores=(2, 0.5, 0, 0))
    prepare(owner)
    assert not owner._live_share_scores
    assert not validate({})['b3_share_score_read']
    user = parse_user_config(dict(device_name='試験', b3_share_score_read=True), tmp_path)
    assert build_pipeline_config(user)['b3_share_score_read']
    with pytest.raises(ValueError):
        validate(dict(b3_share_score_read=1))
