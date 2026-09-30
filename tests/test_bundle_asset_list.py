"""配布資産リスト (packaging/bundle_assets.txt) と score_zero の OCR 代替経路の試験 (2026-09-30)。

user 決定: ゲーム画面の切り出し画像は再配布しない。ui_templates はディレクトリ丸ごとでなく明示列挙とし、
実行時に読まれない画像 (chain_count_digits / ojama / win_panel / *.bak) を含めない。
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
# 配布しない (実行時に読まれないことを imread-trace で確認済み)
FORBIDDEN_PARTS = ('chain_count_digits', 'ojama', 'win_panel', '.bak')
SCORE_DIGIT_COUNT = 10
X_MARK_MIN = 6


def ui_entries() -> list[str]:
    return [line for line in read_asset_list() if line.startswith(UI_DIR)]


def test_ui_templates_are_explicit_files_not_a_directory() -> None:
    entries = ui_entries()
    assert entries, 'ui_templates の記載が無い (score_digits は認識に必須)'
    assert UI_DIR not in entries, 'ディレクトリ丸ごとの記載が残っている'
    assert all(entry.endswith('.png') for entry in entries)


def test_unused_game_images_are_not_listed() -> None:
    for entry in ui_entries():
        assert not any(part in entry for part in FORBIDDEN_PARTS), entry


def test_score_ocr_templates_are_listed() -> None:
    from src.score_ocr import FORMULA_MULT_TEMPLATE_FILENAME
    entries = set(ui_entries())
    for digit in range(SCORE_DIGIT_COUNT):
        assert f'{UI_DIR}/score_digits/digit_{digit}.png' in entries
    assert f'{UI_DIR}/score_digits/{FORMULA_MULT_TEMPLATE_FILENAME}' in entries


def test_x_mark_templates_listed_cover_the_asset_dir_glob() -> None:
    """UiMaskMatcher は x_mark*.png を glob で読む。資産側に増えたのにリストへ足し忘れたら失敗させる。"""
    from src.ui_mask import DEFAULT_TEMPLATE_DIR, X_MARK_GLOB_PATTERN
    listed = {Path(entry).name for entry in ui_entries() if Path(entry).name.startswith('x_mark')}
    assert len(listed) >= X_MARK_MIN
    if not DEFAULT_TEMPLATE_DIR.is_dir():
        pytest.skip('資産ディレクトリが作業ツリーに無い')
    assert {path.name for path in DEFAULT_TEMPLATE_DIR.glob(X_MARK_GLOB_PATTERN)} <= listed


def test_listed_files_exist_when_asset_dir_is_present() -> None:
    if not Path(UI_DIR).is_dir():
        pytest.skip('資産ディレクトリが作業ツリーに無い')
    missing = [entry for entry in ui_entries() if not Path(entry).is_file()]
    assert not missing, missing


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
    from src.recognition_pipeline import _build_score_zero_detector
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
