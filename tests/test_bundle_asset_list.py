"""配布資産リスト (packaging/bundle_assets.txt) と score_zero の OCR 代替経路の試験 (2026-09-30)。

0.6.6 (2026-11-14) からゲーム画面の切り抜き画像 (models/ui_templates) を配布しない。UI の判定は画像なし方式の
決定木 (models/image_free) で行う (DISTRIBUTION_IMAGE_FREE_ALL)。0.6.5 までの明示列挙の試験は git の履歴を参照。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))

from build_bundle import ASSET_LIST, read_asset_list  # noqa: E402
from src.score_zero import (  # noqa: E402
    FROM_OCR_ENV, OcrScoreZeroDetector, ScoreZeroDetector, score_zero_from_ocr_enabled,
)

UI_DIR = 'models/ui_templates'
IMAGE_FREE_MODELS = ('models/image_free/ui.joblib', 'models/image_free/digit_student.joblib',
                     'models/image_free/ui_mask.joblib')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp')


def ui_entries() -> list[str]:
    return [line for line in read_asset_list() if line.startswith(UI_DIR)]


def test_game_screen_images_are_not_listed() -> None:
    """切り抜き画像は 1 枚も同梱しない (ディレクトリ丸ごとの記載も無い)。"""
    assert ui_entries() == []
    assert not [line for line in read_asset_list()
                if line.startswith('models/') and line.lower().endswith(IMAGE_SUFFIXES)]


def test_image_free_models_are_listed_and_distribution_enables_them() -> None:
    from src.production_config import DISTRIBUTION_IMAGE_FREE_ALL, RELEASE_LIVE_ENV_DEFAULTS
    assert set(IMAGE_FREE_MODELS) <= set(read_asset_list())
    assert DISTRIBUTION_IMAGE_FREE_ALL and RELEASE_LIVE_ENV_DEFAULTS['PUYO_IMAGE_FREE_ALL'] == '1'


def test_image_free_default_model_paths_are_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    """配布では環境変数で置き場所を渡さないので、コードの既定の置き場所がそのまま同梱一覧に載っていること
    (11/14: × の判定器の既定 ui_mask.joblib と同梱名 mask_model.joblib が食い違い、配布で起動できなかった)。"""
    from src import image_free_digit, image_free_ui, image_free_ui_mask
    for env in (image_free_digit.MODEL_ENV, image_free_ui.MODEL_ENV, image_free_ui_mask.MODEL_ENV):
        monkeypatch.delenv(env, raising=False)
    listed = set(read_asset_list())
    assert image_free_digit.DEFAULT_MODEL.as_posix() in listed
    assert image_free_ui_mask.DEFAULT_MODEL.as_posix() in listed
    seen = []
    monkeypatch.setattr(image_free_ui, 'load', lambda path: seen.append(Path(path).as_posix()) or {'tasks': {}})
    _ = image_free_ui.LearnedTelop.template_count.fget(object.__new__(image_free_ui.LearnedTelop))
    assert seen and seen[0] in listed


def test_b3_assets_do_not_add_ojama_images() -> None:
    import inspect
    from build_bundle import copy_b3_assets
    assert "'models/ui_templates" not in inspect.getsource(copy_b3_assets)


def test_asset_list_file_is_the_one_build_uses() -> None:
    assert ASSET_LIST.name == 'bundle_assets.txt'


# ---- score_zero の OCR 代替 (既定 OFF) ----

class FakeOcr:
    def __init__(self, values: dict[str, tuple[int | None, float]]) -> None:
        self.values, self.calls = values, 0

    def read_side(self, frame: np.ndarray, side: str) -> tuple[int | None, float]:
        self.calls += 1
        return self.values[side]


FRAME = np.zeros((1080, 1920, 3), np.uint8)


@pytest.mark.parametrize('one,two,both', [
    ((0, .9), (0, .9), True),
    ((0, .9), (1200, .9), False),
    ((None, .3), (0, .9), False),  # 読めない = ゼロと認めない (従来のテンプレート不一致と同じ向き)
    ((None, .0), (None, .0), False),
])
def test_ocr_zero_detector_decision(one: tuple, two: tuple, both: bool) -> None:
    result = OcrScoreZeroDetector(FakeOcr({'1P': one, '2P': two})).detect(FRAME)
    assert result.both_zero is both
    assert (result.score_1p, result.score_2p) == (one[1], two[1])


def test_ocr_zero_detector_rejects_invalid_frame_without_reading() -> None:
    ocr = FakeOcr({'1P': (0, 1.), '2P': (0, 1.)})
    assert OcrScoreZeroDetector(ocr).detect(None).both_zero is False  # type: ignore[arg-type]
    assert ocr.calls == 0


def test_from_ocr_flag_default_off_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FROM_OCR_ENV, raising=False)
    assert score_zero_from_ocr_enabled() is False and score_zero_from_ocr_enabled(None) is False
    monkeypatch.setenv(FROM_OCR_ENV, '0')  # 配布ランチャは OFF のとき "0" を渡す
    assert score_zero_from_ocr_enabled() is False
    monkeypatch.setenv(FROM_OCR_ENV, '1')
    assert score_zero_from_ocr_enabled() is True
    assert score_zero_from_ocr_enabled(False) is False  # 明示指定が環境変数に優先


def test_pipeline_selects_detector_by_flag(capsys: pytest.CaptureFixture[str]) -> None:
    import inspect
    from src.recognition_pipeline import RecognitionPipeline
    from src.recognition_pipeline import _build_score_zero_detector
    assert inspect.signature(RecognitionPipeline.load_default).parameters['score_zero_from_ocr'].default is None
    ocr = FakeOcr({'1P': (0, 1.), '2P': (0, 1.)})
    assert isinstance(_build_score_zero_detector(ocr, True), OcrScoreZeroDetector)
    assert _build_score_zero_detector(None, True) is None
    assert 'score_zero(OCR) unavailable' in capsys.readouterr().out  # 黙って無効化しない
    if Path('models/ui_templates/score_zero/zero_1P.png').is_file():
        assert isinstance(_build_score_zero_detector(None, False), ScoreZeroDetector)


def test_telop_detector_without_templates_reports_no_telop(tmp_path: Path) -> None:
    from src.telop_detector import TelopDetector
    detector = TelopDetector.load_default(template_dir=tmp_path)
    assert detector.template_count == 0
    result = detector.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert result.is_visible is False and result.bbox is None


def test_launcher_warns_when_telop_templates_missing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.phase_j.launcher import check_telop_templates
    assert check_telop_templates(tmp_path) == 0
    assert '中央テロップ検出が無効' in capsys.readouterr().err


def test_launcher_reports_learned_telop_without_images(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """配布の環境 (画像なし方式) では、画像が無くてもテロップ検出は有効と報告する (誤った復元の警告を出さない)。"""
    from src.phase_j.launcher import check_telop_templates
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '1')
    import src.image_free_ui as image_free_ui
    monkeypatch.setattr(image_free_ui, 'load', lambda path: {'tasks': {'telop': None, 'match_end': None}})
    assert check_telop_templates(tmp_path) == 2
    captured = capsys.readouterr()
    assert '有効' in captured.out and '配布物を復元' not in captured.err
