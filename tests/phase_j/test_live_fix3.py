"""画面検証だけの境界で保持し、通常評価と期限では必ず解除する。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.phase_j.live_terminal_hold import TerminalDisplayHold
from src.phase_j.live_bridge import RecognitionBridge


def row(game: int, source: str = 'confirmed_death', **kwargs: object) -> dict:
    return dict(game=game, source=source, probability=1., raw_probability=1.,
                captured_at=10., **kwargs)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('reason', ['input_recovery', 'scored', 'unknown', None])
@pytest.mark.parametrize('normal', [False, True])
def test_boundary_policy(enabled: bool, reason: str, normal: bool) -> None:
    hold = TerminalDisplayHold(enabled)
    hold.observe(row(1), 10.)
    following = row(2, 'static' if normal else 'confirmed_death', terminal_boundary_reason=reason)
    original = deepcopy(following)
    hold.observe(following, 11.)
    expected = enabled and reason == 'input_recovery' and not normal
    assert hold.active(11.) == expected
    assert bool(hold.project(following, 11.).get('_terminal_display_held')) == expected
    assert following == original
    assert not hold.active(15.)


@pytest.mark.parametrize('normal_source', ['static', 'G_fe', 'b3', 'S3'])
def test_first_normal_always_clears(normal_source: str) -> None:
    hold = TerminalDisplayHold(True)
    hold.observe(row(1), 10.)
    hold.observe(row(2, terminal_boundary_reason='input_recovery'), 11.)
    assert hold.deadline == 15.
    hold.observe(row(2, normal_source), 11.1)
    assert not hold.active(11.1)


@pytest.mark.parametrize('device', ['other', None])
def test_device_change_clears(device: str | None) -> None:
    hold = TerminalDisplayHold(True)
    hold.input_changed(dict(device_id='OBS', epoch=0, at=0.))
    hold.observe(row(1), 10.)
    hold.input_changed(dict(device_id=device, epoch=1, at=11.))
    assert not hold.active(11.)
    assert not hold.accepts(row(1))


@pytest.mark.parametrize('forced', [False, True])
@pytest.mark.parametrize('scored', [False, True])
@pytest.mark.parametrize('same', [False, True])
def test_boundary_reason(forced: bool, scored: bool, same: bool) -> None:
    bridge = SimpleNamespace(input_boundary_recovery=same, input_epoch=2, input_failure_epoch=2)
    RecognitionBridge.record_terminal_boundary(bridge, 4, forced, scored)
    assert bridge.terminal_boundary == (4, 'scored' if scored else
                                        'input_recovery' if forced and same else 'unknown')


def test_expiry_does_not_rearm() -> None:
    hold = TerminalDisplayHold(True)
    hold.observe(row(1), 10.)
    for game, stamp in ((2, 11.), (3, 14.), (4, 16.)):
        hold.observe(row(game, terminal_boundary_reason='input_recovery'), stamp)
    assert not hold.active(16.)
    assert not hold.armed


@pytest.mark.parametrize('ready', [False, True])
@pytest.mark.parametrize('streak', [1, 3])
def test_input_reason_requires_verified_failure(ready: bool, streak: int) -> None:
    from src.phase_j.live_process import ProcessRecognitionBridge
    bridge = SimpleNamespace(on_hold=lambda value: None, input_epoch=0, input_device_id='OBS')
    ProcessRecognitionBridge._message(bridge, ('hold', dict(phase='no_puyo_screen', epoch=1,
        device_id='OBS', input_diagnostics=dict(transport_ready=ready, verifier_fail_streak=streak,
                                               verifier_fail_limit=3))))
    assert getattr(bridge, 'input_failure_epoch', None) == (1 if ready and streak == 3 else None)
    assert bridge.input_boundary_recovery


def test_real_loop_records_reason_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import ast
    from tests.phase_j.test_live_forced_boundary import real_generate_loop
    monkeypatch.setenv('PUYO_TERMINAL_HOLD_INPUT_RECOVERY', '1')
    code = ast.unparse(real_generate_loop(True))
    assert '_live_bridge.record_terminal_boundary(game_idx, _forced_boundary, _scored_boundary)' in code
    assert code.index('game_idx += 1') < code.index('_live_bridge.record_terminal_boundary')
    monkeypatch.setenv('PUYO_TERMINAL_HOLD_INPUT_RECOVERY', '0')
    assert 'record_terminal_boundary' not in ast.unparse(real_generate_loop(True))


@pytest.mark.parametrize('enabled', [False, True])
def test_latest_reuses_async_without_changing_other_options(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from src.phase_j.live_speed import configure, SPEED_DEFAULTS
    options = argparse.Namespace(**dict(SPEED_DEFAULTS, live_latest_evaluation=enabled), cpu_threads=1,
        source='dshow', b3_judgment=True, split_evaluation=True, worker_mode='process', compare=False, b3_async=False)
    monkeypatch.setattr('os.environ', {})
    configure(options, argparse.ArgumentParser())
    assert options.b3_async is enabled


@pytest.mark.parametrize('field,value', [('b3_judgment', False), ('split_evaluation', False),
                                        ('worker_mode', 'thread'), ('compare', True)])
def test_latest_rejects_unsupported_mode(field: str, value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    from src.phase_j.live_speed import configure, SPEED_DEFAULTS
    options = argparse.Namespace(**dict(SPEED_DEFAULTS, live_latest_evaluation=True), cpu_threads=1,
        source='dshow', b3_judgment=True, split_evaluation=True, worker_mode='process', compare=False)
    setattr(options, field, value)
    monkeypatch.setattr('os.environ', {})
    with pytest.raises(SystemExit):
        configure(options, argparse.ArgumentParser())


@pytest.mark.parametrize('isolate', [False, True])
@pytest.mark.parametrize('forced', [False, True])
@pytest.mark.parametrize('scored', [False, True])
def test_reason_instrumentation_preserves_boundaries(isolate: bool, forced: bool, scored: bool) -> None:
    import ast
    from src.phase_j.live_bridge import forced_boundary_statements
    states = []
    for enabled in (False, True):
        before, after = forced_boundary_statements(isolate, enabled)
        bridge = SimpleNamespace(consume_input_boundary=lambda: forced,
                                 record_terminal_boundary=lambda *args: None)
        namespace = dict(_live_bridge=bridge, _formal_boundary=scored,
                         _last_formal_boundary_t=1., game_idx=0)
        body = before + ast.parse('if _formal_boundary:\n    game_idx += 1\n'
                                  '    _last_formal_boundary_t = 9.').body + after
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<境界>', 'exec'), namespace)
        states.append((namespace['game_idx'], namespace['_last_formal_boundary_t']))
    assert states[0] == states[1]
