"""撃った盤の旗、初段確定、後段の不一致、EX入力配送を検査する。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.b3_chain_prediction import fire_board_steps
from src.b3_state import SideState
from src.fire_board import RawHistory, clean
from src.phase_j import live_fire_board_config as config
from src.phase_j.live_ex_capture import CaptureEx
from tests.test_b3_chain_prediction import formula, option

SHAPE = (13, 6)


def boards() -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros(SHAPE, dtype=np.int8)
    before[12, :3] = 1
    before[10:13, 3] = 2
    before[9, 3] = 1
    fired = before.copy()
    fired[11:13, 4] = 2
    return before, fired


def state(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> SideState:
    before, fired = boards()
    monkeypatch.setattr('src.b3_state.origin_candidates', lambda *_: ([option((100, 740))], ''))
    side = SideState('1P', enable_fire_board=enabled)
    side.update_frame(.9, dict(state='STABLE', confirmed_board=before, raw_cnn_board=fired), True)
    side.on_fire_start(dict(fire_id='a', trigger_sec=1., observed_at_sec=1., chain_before=before.tolist()))
    formula(side, 1., 100)
    return side


def test_pair_completes_two_stage_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """前の撃った盤の扱い (PUYO_B3_FORECAST=0)。依頼 A の連鎖の予測の扱いは tests/test_live_forecast.py。"""
    monkeypatch.setenv('PUYO_B3_FORECAST', '0')
    before, fired = boards()
    assert fire_board_steps(before) == []
    assert fire_board_steps(fired) == [100, 320]
    off, on = state(False, monkeypatch), state(True, monkeypatch)
    assert off.episode.pred_score == 740
    assert not getattr(off.episode, 'fire_board_tried', False)
    assert len(off.raw_hist.frames) == 0
    assert on.episode.pred_score == 420 and on.episode.pred_l == 2
    assert on.episode.prediction.available_at == 1.01
    assert np.array_equal(on.episode.fire_board.board, fired)
    formula(on, 2., 640)
    assert on.episode.fire_board_conflict
    assert on.episode.pred_score == 740


def test_raw_history_newest_six_and_hidden_row() -> None:
    _, fired = boards()
    fired[0] = [1, 2, 3, 4, 9, 10]
    assert clean(fired)[0].tolist() == [1, 2, 3, 4, 9, 0]
    history = RawHistory()
    for index in range(7):
        history.push(float(index), fired if index == 0 else None)
    assert history.find(7., 100, lambda _: [100]) is None
    history.push(7., fired)
    assert history.find(7., 100, lambda _: [100]) is None
    assert history.find(8., 100, lambda _: [100]).back == 1


def test_flag_defaults_and_worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    opt = parser.parse_args([])
    assert not opt.live_b3_fire_board
    monkeypatch.setenv(config.ENV_ENABLED, '1')
    config.configure(opt, parser)
    assert not SideState('1P').enable_fire_board
    config.configure(parser.parse_args(['--live-b3-fire-board']), parser)
    assert SideState('1P').enable_fire_board
    assert not config.user_value({})
    with pytest.raises(ValueError):
        config.user_value(dict(live_b3_fire_board='true'))


@pytest.mark.parametrize('enabled', [False, True])
def test_launcher_preserves_flag(tmp_path: Path, enabled: bool) -> None:
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    from tests.test_live_24h_launcher import data
    user = parse_user_config(data(tmp_path, live_b3_fire_board=enabled), tmp_path)
    assert user.live_b3_fire_board is enabled
    assert build_pipeline_config(user).get('live_b3_fire_board', False) is enabled


@pytest.mark.parametrize('enabled', [False, True])
def test_ex_capture_only_adds_fire_inputs_with_retr(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_24H_RETR', str(int(enabled)))
    _, fired = boards()
    side = dict(state=dict(state='STABLE'), next_pair=[2, 2], dnext_pair=[1, 1])
    observed = dict(raw_cnn_board=fired.tolist(), formula=dict(valid=True, product=100, left=100))
    row = dict(game_idx=1, _v49f=dict(raw=[side, side]), sides=[observed, observed])
    original = deepcopy(row)
    packet = CaptureEx().capture(None, SimpleNamespace(media_sec=1.), row)
    assert row == original
    for raw in packet['raw']:
        if enabled:
            assert raw['raw_cnn_board'] == fired.tolist()
            assert raw['formula'] == dict(valid=True, product=100)
        else:
            assert 'raw_cnn_board' not in raw and 'formula' not in raw
