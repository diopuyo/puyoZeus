"""元raw score helper/collector/metadataを人工update駆動で検査。実CNNは対象外。"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from typing import Any

import pytest
from scripts import g3_admission as A
from tests.test_g3_metadata_scope import T, B
from tests.test_g3_observer_scope import owned_replace

collector = T.collector


@pytest.fixture(autouse=True)
def admission_root(monkeypatch: Any, tmp_path: Path) -> Path:
    """採録先の根を試験用の一時領域へ差し替える (2026-09-17、W45)。

    `Admission.__init__` は出力が verify の根の下にあることを要求する
    (`scripts/g3_admission.py` の `output_root`)。これは本番の安全弁なので
    **緩めない**。根そのものを試験用へ差し替えて、契約を生かしたまま通す。
    実物の `D:/puyo_analyzer/verify` へは1バイトも書かない。

    これが無かったため、この契約を通る試験は**一度も走っていなかった**
    (実測: 38件が `g3_admission:output_root` で失敗)。
    """
    monkeypatch.setattr(A, "VERIFY", tmp_path)
    return tmp_path


def test_output_outside_the_root_is_rejected(tmp_path: Path) -> None:
    """根の外へは書かせない。差し替えても契約そのものは生きている。"""
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    with pytest.raises(ValueError, match="g3_admission:output_root"):
        A.Admission(outside, "g3-artificial-ui-source")


class Ocr:
    """画像認識値だけを人工入力。ScoreTrackerとhelper自体は元実装。"""
    def __init__(self) -> None:
        self.values: dict[str, int | None] = dict(zip(A.SIDES, (None, None)))

    def read_side(self, image: Any, side: str) -> tuple[int | None, float]:
        return self.values[side], 1.0


def pipeline(c: Any, monkeypatch: Any, *, duplicate: bool = False,
             failure: bool = False) -> tuple[Any, Ocr]:
    cls = c.RecognitionPipeline
    from src.score_ocr import ScoreTracker as tracker
    pipe, ocr = cls.__new__(cls), Ocr()
    pipe._score_tracker_1p, pipe._score_tracker_2p = (tracker(side, ocr) for side in A.SIDES)
    def driver(self: Any, frame_idx: int, time_sec: float, frame: Any) -> object:
        for side in A.SIDES:
            self._update_score_tracker(getattr(self, '_score_tracker_' + side.lower()), frame)
        if duplicate:
            self._update_score_tracker(self._score_tracker_1p, frame)
        if failure:
            raise LookupError('original_update_error')
        return frame
    monkeypatch.setattr(cls, 'update', driver)
    return pipe, ocr


def test_unknown_then_observed_real_metadata(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    c = collector
    pipe, ocr = pipeline(c, monkeypatch)
    state, acc, shared = dict(output=tmp_path), c._LeanNpzAccumulator(), c._SharedGameCounter()
    sides = {s: c._SideState() for s in A.SIDES}
    before = c._process_side_lean
    with ExitStack() as stack:
        B.install(stack, c, None, state, enabled=True)
        observer = A.install(stack, c, state, owned_replace(), source_id='g3-artificial-ui-source')
        for frame in (B.O.FIRST, B.O.FIRST + 2):
            ocr.values['1P'] = None if frame == B.O.FIRST else 90
            image = object()
            assert pipe.update(frame, frame / 60, image) is image
            for side in A.SIDES:
                T.call(c, acc, sides[side], shared, frame, side)
            assert all((s.last_emitted_grid is None) == (frame == B.O.FIRST) for s in sides.values())
        assert observer.withheld == 2 and observer.decisions == 4
    rows = [json.loads(x) for x in (tmp_path / B.O.ENTRIES).read_text().splitlines()]
    count = 0
    for i, row in enumerate(rows):
        count = B.O.validate_row(row, B.O.FIRST + i // 2 * 2, A.SIDES[i % 2], count)
    assert len(rows) == 4 and count == 2 and c._process_side_lean is before
    status = json.loads((tmp_path / 'G3_UI_ADMISSION_STATUS.json').read_text())
    assert status['closed'] and status['references_restored'] and status['error'] is None


@pytest.mark.parametrize('score', [0, 80])
def test_current_raw_not_last_score(collector: Any, monkeypatch: Any, tmp_path: Path, score: int) -> None:
    c = collector
    pipe, ocr = pipeline(c, monkeypatch)
    with ExitStack() as stack:
        observer = A.install(stack, c, dict(output=tmp_path), owned_replace(), source_id='g3-artificial-ui-source')
        ocr.values['1P'] = score
        pipe.update(0, 0.0, object())
        assert observer.latest['status'] == A.OBSERVED
        ocr.values['1P'] = None
        pipe.update(2, 2 / 60, object())
        assert pipe._score_tracker_1p.last_score == score
        assert observer.latest['status'] == A.UNKNOWN
        with pytest.raises(ValueError, match='admission_frame'):
            observer.decide(0, 0.0, '1P', True)


def test_duplicate_read_error_and_restore(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    c = collector
    pipe, ocr = pipeline(c, monkeypatch, duplicate=True)
    original = vars(c.RecognitionPipeline)['_update_score_tracker']
    with pytest.raises(ValueError, match='duplicate_score_read'):
        with ExitStack() as stack:
            A.install(stack, c, dict(output=tmp_path), owned_replace(), source_id='g3-artificial-ui-source')
            pipe.update(0, 0.0, object())
    assert vars(c.RecognitionPipeline)['_update_score_tracker'] is original
    status = json.loads((tmp_path / 'G3_UI_ADMISSION_STATUS.json').read_text())
    assert status['closed'] and status['references_restored'] and 'duplicate_score_read' in status['error']


def test_save_failure_preserves_original_exception(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    c = collector
    pipe, ocr = pipeline(c, monkeypatch, failure=True)
    with pytest.raises(LookupError, match='original_update_error'):
        with ExitStack() as stack:
            observer = A.install(stack, c, dict(output=tmp_path), owned_replace(), source_id='g3-artificial-ui-source')
            observer.stream.close()
            pipe.update(0, 0.0, object())
    status = json.loads((tmp_path / 'G3_UI_ADMISSION_STATUS.json').read_text())
    assert status['closed'] and status['save_error'] and 'original_update_error' in status['error']


@pytest.mark.parametrize('original_error', [False, True])
def test_foreign_descriptor_preserved(original_error: bool) -> None:
    class Owner:
        method = staticmethod(lambda: None)
    foreign = staticmethod(lambda: 1)
    expected = LookupError if original_error else RuntimeError
    with pytest.raises(expected):
        with ExitStack() as stack:
            A.replace_static(stack, Owner, 'method', staticmethod(lambda: 2))
            Owner.method = foreign
            if original_error:
                raise LookupError('original')
    assert vars(Owner)['method'] is foreign


def test_foreign_caller_preserves_physics_updates(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    c = collector
    pipeline(c, monkeypatch)
    state = c._SideState(legit_transition_pending=True)
    grid = [[0] * 6 for _ in range(13)]
    grid[-1][0] = 1
    with ExitStack() as stack:
        observer = A.install(stack, c, dict(output=tmp_path), owned_replace(), source_id='g3-artificial-ui-source')
        c._should_emit(state, c.Board.from_list(grid), c.BoardState.STABLE,
                       enable_physics_persistence_filter=True, physics_sim=c.ChainSimulator())
        assert state.legit_transition_pending is False and observer.decisions == 0
