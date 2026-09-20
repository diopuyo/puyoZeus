"""既知画像→元OCR/helper→採録資格の接続。連続CNN/GTの検査ではない。"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

import pytest
from tests.test_g3_admission import A, collector, owned_replace, pipeline, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)

SOURCE = Path(__file__).resolve().parents[1] / 'data/frames/video_38.mp4'
FROZEN = SOURCE.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'


@pytest.mark.parametrize('frame,expected', [(0, False), (4, False), (3812, False), (8874, True)])
def test_actual_score_to_admission(collector: Any, monkeypatch: Any, tmp_path: Path,
                                 frame: int, expected: bool) -> None:
    import cv2
    from src.score_ocr import ScoreOcr, ScoreTracker
    pipe, _ = pipeline(collector, monkeypatch)
    ocr = ScoreOcr.load_default(FROZEN / 'models/ui_templates/score_digits')
    pipe._score_tracker_1p, pipe._score_tracker_2p = (ScoreTracker(side, ocr) for side in A.SIDES)
    cap = cv2.VideoCapture(str(SOURCE))
    try:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok and cap.get(cv2.CAP_PROP_POS_FRAMES) == frame + 1
    finally:
        cap.release()
    assert not cap.isOpened()
    with ExitStack() as stack:
        observer = A.install(stack, collector, dict(output=tmp_path), owned_replace(),
                             source_id='isolated-image-fixture-not-continuous-run')
        pipe.update(frame, frame / 60, image)
        assert observer.decide(frame, frame / 60, '1P', True) is expected
        assert observer.decide(frame, frame / 60, '2P', True) is expected
        if expected:
            assert observer.latest['scores'] == {'1P': 0, '2P': 0}
    status = json.loads((tmp_path / 'G3_UI_ADMISSION_STATUS.json').read_text())
    assert status['closed'] and status['references_restored'] and status['error'] is None


def test_missing_board_short_circuits_before_admission(collector: Any, monkeypatch: Any,
                                                      tmp_path: Path) -> None:
    """MENU/盤面欠測は元collectorが先に返り、資格判定回数には含まれない。"""
    from tests.test_g3_admission import B
    c = collector
    pipe, _ = pipeline(c, monkeypatch)
    state = dict(output=tmp_path)
    acc, shared = c._LeanNpzAccumulator(), c._SharedGameCounter()
    with ExitStack() as stack:
        B.install(stack, c, None, state, enabled=True)
        observer = A.install(stack, c, state, owned_replace(), source_id='artificial-missing-board')
        frame = B.O.FIRST
        pipe.update(frame, frame / 60, object())
        for side in A.SIDES:
            c._process_side_lean(acc, c._SideState(), side, None, c.BoardState.MENU,
                                 None, 'artificial', frame / 60, frame, shared_game=shared)
        assert observer.frames == 1 and observer.decisions == 0
    rows = [json.loads(line) for line in (tmp_path / B.O.ENTRIES).read_text().splitlines()]
    assert len(rows) == 2 and all(row['arguments']['board'] is None and not row['appends'] for row in rows)
