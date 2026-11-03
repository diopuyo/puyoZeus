"""v40m7 の誤読・早期終了・起点ずれと、別連鎖の境界を小さく再現する。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from src.b3_state import Episode, SideState
from src.chain_continuity import ChainBoundary
from src.fire_board import ChainForecast

BOARD = np.array([
    [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 5, 3], [2, 4, 2, 3, 3, 3],
    [3, 3, 3, 4, 5, 2], [4, 4, 4, 5, 5, 9], [5, 3, 3, 4, 4, 9],
    [9, 2, 3, 5, 3, 2], [9, 2, 2, 9, 9, 4], [5, 9, 9, 9, 9, 4],
    [5, 9, 9, 9, 5, 3], [3, 4, 3, 5, 2, 5], [3, 3, 4, 3, 3, 5],
    [4, 4, 3, 2, 2, 2]], dtype=np.int8)
SCORES = (40, 320, 1140, 2100, 2560)
FLAGS = ('live_chain_forecast_recheck', 'live_chain_end_formula')


def enable(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in FLAGS:
        monkeypatch.setenv('PUYO_' + flag.upper(), '1')


def event(t: float, product: int) -> dict:
    return dict(t_sec=t, observed_at_sec=t, erased=4, product=product)


def test_first_raw_formula_starts_before_accumulated_event(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    side.update_frame(791.133, dict(state='CHAIN', score=None,
                                  formula=dict(valid=True, product=40)), True)
    assert side.episode is not None and side.episode.active
    assert side.episode.first_trigger == 791.133
    side.on_formula(event(791.133, 40))
    assert len(side.episode.steps) == 1


def test_score_start_uses_last_read_across_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    side.update_frame(925.0, dict(state='STABLE', score=100), True)
    side.update_frame(925.1, dict(state='CHAIN', score=None), True)
    side.update_frame(925.233, dict(state='CHAIN', score=140), True)
    assert side.episode is not None and side.episode.active
    assert side.episode.first_trigger == 925.233


def test_confirmed_match_end_closes_without_a_next_piece(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from src.b3_state import GameState
    enable(monkeypatch)
    gs = GameState(SimpleNamespace(rows_of=lambda side: ()))
    side = dict(state='CHAIN', score=100, formula=dict(valid=True, product=40))
    frame = dict(frame=1, t_sec=10, available_at_sec=10, game_idx=0,
                 match_active=True, match_end_locked=False, sides=[side, dict(side)])
    gs.update_frame(frame)
    episodes = [s.episode for s in gs.sides]
    assert all(ep is not None and ep.active for ep in episodes)
    menu = dict(state='MENU', score=100)
    gs.update_frame(dict(frame, frame=2, t_sec=12, available_at_sec=12,
                         match_active=False, match_end_locked=True, sides=[menu, dict(menu)]))
    for st, ep in zip(gs.sides, episodes):
        st.on_formula(event(11.9, 320))
        assert not ep.active and ep.closed_t == 12
        assert len(st.closed) == 1


def test_recent_score_rejects_midchain_next_candidate() -> None:
    boundary = ChainBoundary()
    initial = dict(next_pair=(1, 2), dnext_pair=(3, 4), score=2619)
    boundary.observe(266, initial)
    boundary.observe(266.033, initial)
    shifted = dict(next_pair=(3, 4), dnext_pair=(2, 2), score=4319, next_slide_motion=True)
    boundary.observe(267.266, shifted)
    boundary.observe(267.3, shifted)
    assert boundary.t == float('-inf')


@pytest.mark.parametrize('kind', ['placed', 'unchanged', 'recent_score'])
def test_stale_activity_needs_two_supported_placed_cells(kind: str) -> None:
    boundary = ChainBoundary()
    old = np.zeros((13, 6), dtype=np.int8)
    old[12, 0] = 1
    side = dict(confirmed_board=old, raw_cnn_board=old, score=100, tsumo_count=1)
    boundary.observe(0, dict(side, formula=dict(valid=True, product=40)))
    boundary.observe(.033, side)
    new = old.copy()
    if kind != 'unchanged':
        new[12, 1:3] = (2, 3)
    if kind == 'recent_score':
        boundary.observe(2.4, dict(side, score=140))
    placed = dict(side, confirmed_board=new, raw_cnn_board=new, tsumo_count=2, landing_diag={'source':'test'})
    boundary.observe(2.5, placed)
    boundary.observe(2.533, dict(placed, landing_diag=None))
    assert (boundary.kind == 'placement') == (kind == 'placed')


@pytest.mark.parametrize('frozen', [False, True])
def test_confirmed_same_placement_rechecks_first_formula(monkeypatch: pytest.MonkeyPatch, frozen: bool) -> None:
    enable(monkeypatch)
    if frozen:
        from src.phase_j._live_ex_frozen.src.fire_board import ChainForecast as Forecast
    else:
        Forecast = ChainForecast
    forecast = Forecast()
    wrong = BOARD.copy()
    wrong[1, 4] = wrong[3, 4] = 9
    forecast.observe(64.2, None, BOARD)
    forecast.observe(64.233, None, wrong)
    forecast.observe(64.267, dict(valid=True, product=40), BOARD)
    assert forecast.current().steps == (40,)  # 確定発火盤の到着までは裏付け不足。
    side = dict(chain=dict(trigger_sec=64.3, before_board=BOARD.tolist()))
    forecast.observe(64.3, dict(valid=True, product=40), BOARD, side)
    assert forecast.current().steps == SCORES
    assert forecast.current().chain_id == 64.267


@pytest.mark.parametrize('kind', ['stale', 'hidden', 'erasing', 'unconfirmed'])
def test_recheck_does_not_select_longest(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    enable(monkeypatch)
    wrong = BOARD.copy()
    wrong[1, 4] = wrong[3, 4] = 9
    forecast = ChainForecast()
    forecast.observe(10, None, wrong)
    forecast.observe(10.03, dict(valid=True, product=40), wrong)
    candidate = BOARD.copy()
    side = dict(chain=dict(trigger_sec=10.03, before_board=BOARD.tolist()))
    if kind == 'stale':
        side['chain']['trigger_sec'] = 8
    elif kind == 'hidden':
        candidate[0, 0] = 4
    elif kind == 'erasing':
        candidate[1:4] = 0
    else:
        side = {}
    forecast.observe(10.06, dict(valid=True, product=40), candidate, side)
    forecast.observe(10.09, dict(valid=True, product=40), candidate, side)
    assert forecast.current().steps == (40,)


def test_stable_and_state_name_do_not_end_formula_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P', enable_fire_board=True)
    side.on_formula(event(64.267, 40))
    episode = side.episode
    for t, state in [(64.3, 'CHAIN'), (65.3, 'STABLE'), (65.333, 'STABLE'), (66.3, 'OJAMA_FALL')]:
        side.update_frame(t, dict(state=state, score=40), True)
    assert episode.active
    side.on_formula(event(65.583, 320))
    side.on_formula(event(66.967, 1140))
    side.on_fire_start(dict(fire_id='old', trigger_sec=65.5, observed_at_sec=68.7, chain_before=BOARD))
    assert side.episode is episode
    assert episode.first_trigger == 64.267
    assert [s[2] for s in episode.steps] == [40, 320, 1140]


def test_score_recovers_same_episode_without_losing_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    side.on_formula(event(64.267, 40))
    side.episode.sig_stable_t = side.episode.closed_t = 65.333
    side.update_frame(65, dict(state='STABLE', score=0), True)
    side.update_frame(65.167, dict(state='STABLE', score=40), True)
    assert side.episode.active
    assert side.episode.first_trigger == 64.267
    assert len(side.episode.steps) == 1


def test_confirmed_next_separates_short_chains(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    initial = dict(state='STABLE', next_pair=(1, 2), dnext_pair=(3, 4))
    for t in (9.9, 9.93):
        side.update_frame(t, initial, True)
    side.on_formula(event(10, 40))
    old = side.episode
    advanced = dict(state='STABLE', next_pair=(3, 4), dnext_pair=(2, 2), next_slide_motion=True)
    for t in (10.8, 10.83):
        side.update_frame(t, advanced, True)
    assert not old.active
    assert old.sig_next_t == 10.83
    side.on_formula(event(11, 40))
    assert side.episode is not old and side.episode.active
    assert side.episode.first_trigger == 11
    assert len(side.episode.steps) == 1


def test_actual_ojama_increase_requires_board_support() -> None:
    boundary = ChainBoundary()
    empty = np.zeros((13, 6), dtype=np.int8)
    for t in (0, .03):
        boundary.observe(t, dict(confirmed_board=empty, raw_cnn_board=empty))
    boundary.observe(1, dict(state='OJAMA_FALL', next_slide_motion=True))
    assert boundary.t == float('-inf')
    landed = empty.copy()
    landed[12] = 9
    for t in (2, 2.03):
        boundary.observe(t, dict(confirmed_board=landed, raw_cnn_board=landed))
    assert boundary.kind == 'ojama' and boundary.t == 2.03


def test_next_shift_during_formula_tail_is_confirmed_after_silence() -> None:
    boundary = ChainBoundary()
    for t in (0, .03):
        boundary.observe(t, dict(next_pair=(1, 2), dnext_pair=(3, 4)))
    advanced = dict(next_pair=(3, 4), dnext_pair=(2, 2), next_slide_motion=True)
    for t in (1, 1.03):
        boundary.observe(t, dict(advanced, formula=dict(valid=True, product=40)))
    assert boundary.t == float('-inf')
    boundary.observe(1.54, advanced)
    assert boundary.t == 1.54 and boundary.kind == 'next'


def test_next_flash_and_count_alone_do_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    monkeypatch.setenv('PUYO_CHAIN_SPLIT_PLACE', '1')
    side = SideState('1P')
    side.on_formula(event(10, 40))
    for t, count in [(10.1, 1), (11, 2)]:
        side.update_frame(t, dict(state='STABLE', tsumo_count=count, next_slide_motion=True), True)
    side.on_formula(event(11.1, 320))
    assert side.episode.first_trigger == 10
    assert len(side.episode.steps) == 2


def test_score_only_can_start_an_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    side.update_frame(1, dict(state='STABLE', score=100), True)
    side.update_frame(2, dict(state='STABLE', score=140), True)
    assert side.episode.active and side.episode.first_trigger == 2


def test_baseline_without_formula_or_score_does_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('2P')
    side.update_frame(82.5, dict(state='STABLE', score=31626), True)
    side.on_fire_start(dict(fire_id='false', trigger_sec=82.5, observed_at_sec=82.5, chain_before=BOARD))
    side.update_frame(83.03, dict(state='STABLE', score=31629), True)
    assert side.episode is None
    assert side.pending_fire is not None
    side.on_formula(event(83.1, 40))
    assert side.episode.active and side.episode.first_trigger == 83.1


def test_new_score_evidence_after_boundary_starts_separate_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    side = SideState('1P')
    side.on_formula(event(10, 40))
    old = side.episode
    side.boundary.t, side.boundary.kind = 11, 'next'
    side.update_frame(11, dict(state='STABLE', score=40), True)
    side.on_fire_start(dict(fire_id='next', trigger_sec=12, observed_at_sec=12, chain_before=BOARD))
    side.update_frame(12.1, dict(state='STABLE', score=80), True)
    assert not old.active and side.episode is not old
    assert side.episode.active and side.episode.first_trigger == 12.1


def test_same_color_next_needs_confirmed_placement() -> None:
    boundary = ChainBoundary()
    side = dict(next_pair=(1, 1), dnext_pair=(1, 1), tsumo_count=4)
    for t in (0, .03):
        boundary.observe(t, side)
    boundary.observe(1, dict(side, next_slide_motion=True))
    assert boundary.t == float('-inf')
    boundary.observe(1.1, dict(side, tsumo_count=5, landing_diag=dict(source='next_queue_2')))
    assert boundary.t == 1.1 and boundary.kind == 'next'


def test_formula_gap_without_boundary_preserves_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    enable(monkeypatch)
    forecast = ChainForecast()
    forecast.observe(10, None, BOARD)
    forecast.observe(10.03, dict(valid=True, product=40), BOARD)
    forecast.observe(12, dict(valid=True, product=320), BOARD)
    assert forecast.current(14).chain_id == 10.03
    assert forecast.current().observed == (40, 320)


@pytest.mark.parametrize('key', FLAGS)
def test_flags_default_off_and_spawn_delivery(key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_speed
    from src.phase_j.launcher import UserConfig, build_pipeline_config, parse_user_config
    from src.phase_j.release_config import release_user_config
    assert live_speed.SPEED_DEFAULTS[key] is False
    assert UserConfig.__dataclass_fields__[key].default is False
    assert release_user_config({}, tmp_path)[key] is False
    assert release_user_config({key: True}, tmp_path)[key] is True
    user = parse_user_config(dict(source='dshow', device_name='OBS', **{key: True}), tmp_path)
    assert build_pipeline_config(user)[key] is True
    parser = argparse.ArgumentParser()
    live_speed.add_arguments(parser)
    options = parser.parse_args(['--' + key.replace('_', '-')])
    options.output, options.source, options.cpu_threads = tmp_path, 'dshow', 1
    monkeypatch.setattr('os.environ', {})
    live_speed.configure(options, parser)
    assert live_speed.trial_flags_record(options)['chain_continuity'][key] == dict(enabled=True, env='1')
