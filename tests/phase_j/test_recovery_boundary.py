"""復帰境界の統合は番号だけを変え、死亡状態の初期化を省かない。"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.replay_forced_boundary_20261011 import run_mini_generate
from src.phase_j.live_boundary import RecoveryBoundary, RECOVERY_BOUNDARY_WINDOW_SEC
from src.phase_j.live_bridge import coalesce_boundary_block
from src.phase_j.launcher import parse_user_config, build_pipeline_config, LauncherConfigError


@pytest.mark.parametrize('gap', [0.01, 1., 3.81, 4.999])
def test_score_reset_runs_without_a_second_game(gap: float) -> None:
    events = [(10., (20000, 10000), True), (10.+gap, (0, 0), False)]
    trace = run_mini_generate(events, True, coalesce=True)
    assert [game for _, game, _ in trace] == [1, 1]
    assert trace[-1][2] == 10.+gap  # 得点境界の本体を通ってデバウンス枠を更新する。
    assert run_mini_generate(events, True)[-1][1] == 2


@pytest.mark.parametrize('gap', [RECOVERY_BOUNDARY_WINDOW_SEC, 6., 30.])
def test_distant_score_reset_is_another_game(gap: float) -> None:
    state = RecoveryBoundary()
    assert state.increment(True, False, 10.) == 1
    assert state.increment(False, True, 10.+gap) == 1


def test_real_device_switch_is_not_coalesced() -> None:
    state = RecoveryBoundary()
    assert state.increment(True, False, 10., recovery=False) == 1
    assert state.increment(False, True, 11.) == 1


def test_only_first_score_boundary_is_coalesced() -> None:
    state = RecoveryBoundary()
    assert state.increment(True, False, 10.) == 1
    assert state.increment(False, True, 11.) == 0
    assert state.increment(False, True, 12.) == 1


def test_simultaneous_and_repeated_forced_boundaries() -> None:
    state = RecoveryBoundary()
    assert state.increment(True, True, 10.) == 1
    assert state.increment(False, True, 11.) == 1
    assert state.increment(True, False, 12.) == 1
    assert state.increment(True, False, 13.) == 1
    assert state.increment(False, True, 14.) == 0


@pytest.mark.parametrize('recovery,expected', [(True, 0), (False, 1)])
def test_second_forced_with_score_reset_uses_pending_recovery(recovery: bool, expected: int) -> None:
    state = RecoveryBoundary()
    assert state.increment(True, False, 10.) == 1
    assert state.increment(True, True, 12., recovery=recovery) == expected
    assert state.forced_at is None


def test_death_reset_and_ending_game_are_preserved() -> None:
    block = ast.parse('if boundary:\n    ending = game_idx\n    game_idx += 1\n    death = None').body[0]
    coalesce_boundary_block(block)
    from types import SimpleNamespace
    namespace = dict(boundary=True, game_idx=5, death='1P', _forced_boundary=False,
                     _scored_boundary=True, t=12.,
                     _live_bridge=SimpleNamespace(boundary_increment=lambda *args: 0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                 '<boundary>', 'exec'), namespace)
    assert namespace['game_idx'] == namespace['ending'] == 5
    assert namespace['death'] is None


@pytest.mark.parametrize('enabled', [True, False])
def test_config_roundtrip(tmp_path: Path, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from scripts.run_live_pipeline_20260928 import parse_args
    config = build_pipeline_config(parse_user_config(
        dict(device_name='OBS', recovery_boundary_coalescing=enabled), tmp_path))
    assert ('recovery_boundary_coalescing' in config) == enabled
    path = tmp_path / 'live.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['live', '--config', str(path)])
    assert parse_args().recovery_boundary_coalescing is enabled


@pytest.mark.parametrize('value', [1, 'true', None])
def test_bad_config_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(LauncherConfigError):
        parse_user_config(dict(device_name='OBS', recovery_boundary_coalescing=value), tmp_path)
