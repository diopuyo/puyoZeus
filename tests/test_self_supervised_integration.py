"""RecognitionPipeline + 擬似ラベル統合テスト.

backwards compat:
    - enable_pseudo_label=False (default) で既存挙動完全維持
    - enable_pseudo_label=True で validator が active 化
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.image_reader import ImageReader
from src.match_state import MatchStateDetector
from src.recognition_pipeline import (
    PipelineResult,
    RecognitionPipeline,
    SideResult,
)
from src.self_supervised.label_store import LabelStore
from src.self_supervised.score_fine_tuner import (
    MIN_SAMPLES_PER_LABEL,
    ScoreFineTuner,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_SCORE,
    PseudoLabelSample,
)


def _make_simple_pipeline(enable_pseudo: bool = False) -> RecognitionPipeline:
    """最小依存で構築."""
    reader = ImageReader(use_match_state=False)
    md = MatchStateDetector.load_default()
    return RecognitionPipeline(
        image_reader=reader,
        match_state_detector=md,
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        next_detector=None,
        enable_pseudo_label=enable_pseudo,
    )


def _frame_1080p() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# backwards compat
# ============================


def test_pipeline_default_no_pseudo_validators():
    """default では enable_pseudo_label=False、validator 無し."""
    pipe = _make_simple_pipeline(enable_pseudo=False)
    assert pipe._enable_pseudo_label is False
    assert pipe._pseudo_validators == []


def test_pipeline_with_pseudo_label_init_validators():
    """enable_pseudo_label=True で全 validator が init される."""
    pipe = _make_simple_pipeline(enable_pseudo=True)
    assert pipe._enable_pseudo_label is True
    # 5 validator (Score / Next / Chain / HiddenRow / CellColor) が登録されている
    assert len(pipe._pseudo_validators) == 5


def test_pipeline_update_compat_no_pseudo():
    """pseudo 無しで update() は既存挙動通り."""
    pipe = _make_simple_pipeline(enable_pseudo=False)
    frame = _frame_1080p()
    res = pipe.update(0, 0.0, frame)
    assert isinstance(res, PipelineResult)
    assert isinstance(res.p1, SideResult)
    # default flush は 0
    assert pipe.flush_pseudo_labels() == 0
    assert pipe.collect_pseudo_labels() == []


def test_pipeline_update_with_pseudo():
    """pseudo 有効で update() が validator を回す (crash 無し)."""
    pipe = _make_simple_pipeline(enable_pseudo=True)
    frame = _frame_1080p()
    for i in range(5):
        res = pipe.update(i, i * 0.1, frame)
        assert isinstance(res, PipelineResult)
    # collect で 0 件以上、crash 無し
    samples = pipe.collect_pseudo_labels()
    assert isinstance(samples, list)


def test_pipeline_flush_to_store(tmp_path: Path):
    """flush_pseudo_labels が LabelStore に書き込む."""
    store = LabelStore(video_id="test_video", root=tmp_path)
    pipe = _make_simple_pipeline(enable_pseudo=True)
    pipe._pseudo_label_store = store
    # 仮の擬似ラベルを直接 buffer に積む
    pipe._pseudo_validators[0]._buffer.append(
        PseudoLabelSample(
            component=COMPONENT_SCORE,
            timestamp=0.0,
            input_data={"patch": np.zeros((50, 40, 3), dtype=np.uint8)},
            label=0,
            confidence=0.95,
            metadata={},
        )
    )
    n = pipe.flush_pseudo_labels()
    assert n == 1
    # store に保存されている
    loaded = list(store.load(COMPONENT_SCORE))
    assert len(loaded) == 1


def test_pipeline_reset_clears_validators():
    """RecognitionPipeline.reset() は基本 state を clear。
    validator も別途 reset 可能であることを確認."""
    pipe = _make_simple_pipeline(enable_pseudo=True)
    # validator に擬似ラベルを直接 emit
    pipe._pseudo_validators[0]._buffer.append(
        PseudoLabelSample(
            component=COMPONENT_SCORE, timestamp=0.0,
            input_data={}, label=0, confidence=0.95,
        )
    )
    # validator 個別 reset で buffer クリア
    for v in pipe._pseudo_validators:
        v.reset()
    assert pipe.collect_pseudo_labels() == []


# ============================
# ScoreFineTuner
# ============================


def test_score_fine_tuner_no_samples(tmp_path: Path):
    """サンプル 0 件で fine_tune は no-op."""
    tuner = ScoreFineTuner(template_dir=tmp_path)
    metrics = tuner.fine_tune([])
    assert metrics["n_samples"] == 0
    assert metrics["n_labels_updated"] == 0


def test_score_fine_tuner_below_min_samples(tmp_path: Path):
    """label あたり MIN_SAMPLES_PER_LABEL 未満なら更新せず."""
    tuner = ScoreFineTuner(template_dir=tmp_path)
    samples = []
    for _ in range(MIN_SAMPLES_PER_LABEL - 1):
        samples.append(PseudoLabelSample(
            component=COMPONENT_SCORE, timestamp=0.0,
            input_data={"patch": np.full((50, 40, 3), 128, dtype=np.uint8)},
            label=0, confidence=0.95,
        ))
    metrics = tuner.fine_tune(samples)
    assert metrics["n_labels_updated"] == 0


def test_score_fine_tuner_writes_template(tmp_path: Path):
    """十分な数のサンプルあれば digit_N.png が更新される."""
    tuner = ScoreFineTuner(template_dir=tmp_path)
    # label=3 を 10 サンプル
    samples = []
    for _ in range(10):
        samples.append(PseudoLabelSample(
            component=COMPONENT_SCORE, timestamp=0.0,
            input_data={"patch": np.full((50, 40, 3), 200, dtype=np.uint8)},
            label=3, confidence=0.95,
        ))
    metrics = tuner.fine_tune(samples)
    assert metrics["n_labels_updated"] == 1
    assert metrics["samples_per_label"][3] == 10
    # ファイルが書かれた
    assert (tmp_path / "digit_3.png").is_file()


def test_score_fine_tuner_rollback(tmp_path: Path):
    """rollback で元のテンプレに戻る."""
    import cv2
    # 元テンプレを書いておく
    orig = np.full((50, 40, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "digit_3.png"), orig)
    tuner = ScoreFineTuner(template_dir=tmp_path)
    samples = [
        PseudoLabelSample(
            component=COMPONENT_SCORE, timestamp=0.0,
            input_data={"patch": np.full((50, 40, 3), 250, dtype=np.uint8)},
            label=3, confidence=0.95,
        ) for _ in range(10)
    ]
    tuner.fine_tune(samples)
    # 更新後テンプレは 50 から離れる
    after = cv2.imread(str(tmp_path / "digit_3.png"))
    assert int(after.mean()) > 60
    # rollback
    tuner.rollback()
    restored = cv2.imread(str(tmp_path / "digit_3.png"))
    assert abs(int(restored.mean()) - 50) <= 1


def test_score_fine_tuner_invalid_lr():
    with pytest.raises(ValueError):
        ScoreFineTuner(lr=0.0)
    with pytest.raises(ValueError):
        ScoreFineTuner(lr=1.5)
