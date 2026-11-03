"""画像禁止、停止監視、差分保存と再開の境界を検証する。"""
from __future__ import annotations

import json
import gzip
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.compare_image_free_all import checkpoint, compare, mode
from src import image_free_ui as ui


def test_checkpoint_repeated_boundary_preserves_differences(tmp_path: Path) -> None:
    state = dict(frame=300, fingerprint='test')
    checkpoint(tmp_path, state, [dict(frame=299)])
    checkpoint(tmp_path, state, [])
    with gzip.open(tmp_path/'000000300.differences.json.gz', 'rt') as stream:
        assert json.load(stream)['rows'] == [dict(frame=299)]
    state['frame'] = 301
    checkpoint(tmp_path, state, [dict(frame=300)])
    manifest = json.loads((tmp_path/'checkpoint.json').read_text())
    assert len(manifest['chunks']) == 2
    assert not (tmp_path/'000000300.checkpoint.pkl').exists()


def test_display_scenes_continue_across_checkpoint() -> None:
    state = dict(counts={}, display_scenes=0, previous_display_difference=False)
    for index, values in enumerate(((.5, .6), (.5, .6), (.5, .5), (.5, .6))):
        compare([dict(displayed_probability=p) for p in values], state, index, 60.)
    assert state['display_scenes'] == 2
    assert state['counts']['displayed_probability'] == 3


def test_candidate_blocks_image_reads() -> None:
    with pytest.raises(RuntimeError, match='画像読込'), mode(True):
        cv2.imread('forbidden.png')


def test_swallowed_image_read_failure_still_fails() -> None:
    with pytest.raises(RuntimeError, match='握りつぶし'):
        with mode(True):
            try:
                cv2.imread('forbidden.png')
            except RuntimeError:
                pass


def test_all_factories_do_not_read_images(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.match_end_detector import MatchEndDetector
    from src.telop_detector import TelopDetector
    from src.score_zero import ScoreZeroDetector
    from src.win_panel import WinPanelDetector
    from src.chain_count_ocr import ChainCountOcr
    from src.exchange_event_terminal import ObservedDeathDetector
    from src.review_terminal_outcome import ReviewTerminalOutcomeDetector
    from src.ojama_warning import OjamaWarningDetector
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '1')
    with mode(True):
        for factory in (MatchEndDetector.load_default, TelopDetector.load_default,
                        ScoreZeroDetector.load_default, WinPanelDetector.load_default,
                        ChainCountOcr.load_default, ReviewTerminalOutcomeDetector.load_default,
                        ObservedDeathDetector):
            factory()
        assert isinstance(OjamaWarningDetector(use_cnn=False)._templates, ui.LearnedTemplates)


def test_terminal_preserves_consecutive_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.exchange_event_terminal import ObservedDeathDetector
    monkeypatch.setenv('PUYO_IMAGE_FREE_ALL', '1')
    monkeypatch.setattr(ui, 'learned_score', lambda key, frame: .56 if key == 'death_0' else .54)
    reader = ObservedDeathDetector()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert reader.update(frame) == ()
    assert reader.update(frame) == ('1P',)
    monkeypatch.setattr(ui, 'learned_score', lambda key, frame: .54)
    assert reader.update(frame) == ()


def test_guard_query_failure_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import image_free_supervisor as supervisor
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError('監視失敗')
    monkeypatch.setattr(supervisor.subprocess, 'run', fail)
    paused, reason = supervisor.live_running()
    assert paused and '監視不能' in reason
