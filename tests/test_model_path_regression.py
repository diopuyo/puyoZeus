"""default CNN model path / cnn_override_prob 回帰防止.

CYCLE_FINDINGS.md 1.-1 「間違ったデフォルトモデル」 罠 (= cycle 20/21 で
default model が古いものに戻った) と、 1.1 cnn_override_prob = 0.70 確定値の保護。
"""
from src.hybrid_classifier import DEFAULT_CNN_OVERRIDE_PROB
from src.recognition_pipeline import RecognitionPipeline


def test_default_cnn_model_path_file_exists() -> None:
    """default model path で指されるファイルが存在すること."""
    path = RecognitionPipeline.DEFAULT_CNN_MODEL_PATH
    assert path.exists(), f"default CNN model not found: {path}"


def test_default_cnn_override_prob_is_0_70() -> None:
    """cnn_override_prob default = 0.70 (= CYCLE_FINDINGS.md 1.1 確定値).

    cycle 検証で「これより緩めても厳しくしても劣化」 と確定済。
    変更する場合は専用 cycle で再確定が必要。
    """
    assert DEFAULT_CNN_OVERRIDE_PROB == 0.70


def test_image_reader_bg_extreme_threshold_is_25() -> None:
    """cycle 37 採用値 bg_extreme_threshold = 25.0 の凍結.

    cycle 33/34/35 で 20/25/27 の sweep を経て 25 で確定。
    """
    from src.image_reader import ImageReader
    reader = ImageReader()
    assert reader._bg_extreme_threshold == 25.0, (
        f"bg_extreme_threshold changed to {reader._bg_extreme_threshold}"
    )
