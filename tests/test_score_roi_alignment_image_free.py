"""得点の欄の位置合わせ: 画像なし方式の OCR では × らしさの判定器で各位置の点数を出す。"""
import numpy as np

from src.score_roi_alignment import CANDIDATES, ScoreRoiAlignment, candidate_scores


class _FakeImageFreeOcr:
    """切り抜きの × を持たない OCR (触ったら失敗する) の代わり。"""

    def __init__(self, best: tuple[int, int]) -> None:
        self.best = best
        self._region_offsets = {'1P': (0, 0), '2P': (0, 0)}

    @property
    def _mult_template_gray(self) -> None:
        raise AssertionError('画像なし方式で切り抜きの × に触った')

    def multiply_scores_at_offsets(self, frame, side, offsets):
        return np.array([0.95 if o == self.best else 0.5 for o in offsets])

    def read_side_detail_at_offset(self, frame, side, dx, dy):
        return 0, 1.0, (4, 0, 10, 6, 4, None, None, None), (1.0,) * 5 + (0.0,) * 3


def test_candidate_scores_use_the_multiply_model_not_the_template() -> None:
    scores = candidate_scores(np.zeros((1080, 1920, 3), np.uint8), '1P', _FakeImageFreeOcr((2, -1)))
    assert scores.shape == (len(CANDIDATES),)
    assert CANDIDATES[int(np.argmax(scores))] == (2, -1)


def test_alignment_settles_on_the_model_offset(monkeypatch) -> None:
    import src.score_roi_alignment as alignment
    monkeypatch.setattr(alignment, 'parse_formula_cells', lambda *a: type('R', (), {'valid': True})())
    ocr = _FakeImageFreeOcr((0, -2))
    align = ScoreRoiAlignment(ocr)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    for k in range(alignment.FORMULA_SAMPLES):
        align.observe(frame, k / alignment.SAMPLE_HZ)
    assert align.ready and ocr._region_offsets == {'1P': (0, -2), '2P': (0, -2)}
