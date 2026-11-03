"""公開番号の分離が内部キーを変更しないことを検証する。"""
import pickle
import ast
from types import SimpleNamespace as NS

import pytest

from src.phase_j.live_match_number import PublicMatchNumber


@pytest.mark.parametrize('forced,scored,recovery,expected', [
    (True, False, True, 0), (False, True, False, 1),
    (True, True, True, 1), (True, False, False, 1),
])
def test_boundary(forced: bool, scored: bool, recovery: bool, expected: int) -> None:
    numbers = PublicMatchNumber()
    numbers.boundary(1, forced, scored, recovery)
    row = dict(game=1, probability=0.25)
    before = pickle.dumps(row)
    assert numbers.decorate(row) == dict(row, public_game=expected)
    assert pickle.dumps(row) == before


def test_recovery_then_score_and_old_row() -> None:
    numbers = PublicMatchNumber()
    numbers.boundary(1, True, False, True)
    numbers.boundary(2, False, True, True)
    assert [numbers.number(i) for i in range(3)] == [0, 0, 1]
    with pytest.raises(ValueError):
        numbers.boundary(2, False, True, True)


@pytest.mark.parametrize('enabled', [False, True])
def test_real_evaluator_death_resets_at_incident_boundaries(
        enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """10/11の3.81秒差を実境界注入と実E16のresetで確認する。"""
    from scripts.replay_forced_boundary_20261011 import MINI_GENERATE, FakeBridge, make_packet
    from scripts.visualize_advantage_overlay import (
        _detect_score_reset, accept_formal_boundary, update_score_reset_latch)
    from src.phase_j.live_bridge import adapt_loop
    from src.phase_j.live_evaluation import SplitExchangeOverlay
    monkeypatch.delenv('PUYO_TERMINAL_HOLD_INPUT_RECOVERY', raising=False)
    events = [(695., (29247, 15423)), (696.10, (29247, 15423)), (699.91, (0, 0))]
    packets = [make_packet(i, t, score) for i, (t, score) in enumerate(events)]
    bridge = FakeBridge(packets, {1})
    numbers = PublicMatchNumber()
    bridge.record_public_boundary = lambda g, f, s: numbers.boundary(g, f, s, True)
    tree = ast.parse(MINI_GENERATE)
    loop = next(n for n in tree.body[0].body if isinstance(n, ast.For))
    adapt_loop(loop, True, public_match_number=enabled)
    namespace = dict(_live_bridge=bridge, _detect_score_reset=_detect_score_reset,
        accept_formal_boundary=accept_formal_boundary, update_score_reset_latch=update_score_reset_latch,
        _drive_ojama=lambda *a, **kw: None, _RESOLVED=NS(_counter_tracker=None))
    exec(compile(ast.fix_missing_locations(tree), '<boundary>', 'exec'), namespace)
    trace = namespace['generate'](None)
    assert [g for _, g, _ in trace] == [0, 1, 2]
    overlay = SplitExchangeOverlay(NS(), lambda *a: None, lambda *a: None, e16=True)
    # モデル推論を省き、実update→_reset→E16.beforeの死亡状態を観測する。
    monkeypatch.setattr(overlay, '_advance', lambda *a: None)
    side = NS(chain_event=None)
    for (frame, game, _), (stamp, _) in zip(trace, events):
        result = NS(p1=side, p2=side, confirmed_dead_sides=('2P',) if frame < 2 else ())
        overlay.update(result, None, None, stamp, game)
        assert overlay._e16.dead_sides == ({'2P'} if frame < 2 else set())
    assert not overlay._death_held
    assert overlay.tracker.source != 'confirmed_death'
    assert [numbers.number(g) for _, g, _ in trace] == ([0, 0, 1] if enabled else [0, 1, 2])


def test_public_number_does_not_keep_terminal_or_hide_waiting() -> None:
    from tests.phase_j.test_live_terminal_hold import publisher, publish, terminal, probability
    owner = publisher()
    owner.public_matches = PublicMatchNumber()
    first = publish(owner, terminal(), 10.)
    assert probability(first) == 1.
    owner.public_matches.boundary(1, True, False, True)
    held = publish(owner, dict(terminal(1), raw_probability=None, match_active=False), 11.)
    assert held['identity']['match_id'] == first['identity']['match_id']
    assert probability(held) is None
    assert held['display']['message'] == '次試合待ち'
    owner.public_matches.boundary(2, False, True, True)
    next_game = publish(owner, dict(terminal(2), raw_probability=None), 12.)
    assert probability(next_game) is None
    assert next_game['identity']['match_id'] == '1'


def test_sendable_terminal_uses_public_key() -> None:
    from unittest.mock import patch
    from tests.phase_j.test_live_terminal_hold import publisher, publish, terminal
    owner = publisher()
    owner.public_matches = PublicMatchNumber()
    owner.public_matches.boundary(1, True, False, True)
    value = publish(owner, terminal(1), 10.)
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=10.1):
        assert owner.server.sendable(value) is not None
    assert owner.terminal.game == 1
    assert value['identity']['match_id'] == '0'


@pytest.mark.parametrize('isolate', [False, True])
@pytest.mark.parametrize('terminal_recovery', [False, True])
def test_real_loop_changes_only_boundary_recording(
        isolate: bool, terminal_recovery: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect
    from scripts.visualize_advantage_overlay import generate
    from src.phase_j.live_bridge import adapt_loop
    monkeypatch.setenv('PUYO_TERMINAL_HOLD_INPUT_RECOVERY', str(int(terminal_recovery)))
    loops = []
    for enabled in (False, True):
        tree = ast.parse(inspect.getsource(generate))
        loop = next(n for n in tree.body[0].body if isinstance(n, ast.For)
                    and ast.unparse(n.target) == 'fi')
        adapt_loop(loop, isolate, public_match_number=enabled)
        if enabled:
            loop.body = [n for n in loop.body if 'record_public_boundary' not in ast.unparse(n)]
            if not isolate and not terminal_recovery:
                loop.body = [n for n in loop.body if not ast.unparse(n).startswith(
                    ('_scored_boundary =', '_forced_boundary ='))]
                for n in loop.body:
                    if ast.unparse(n) == '_formal_boundary = _forced_boundary or _scored_boundary':
                        n.value = ast.parse('_live_bridge.consume_input_boundary() or _formal_boundary').body[0].value
        loops.append(ast.dump(loop))
    assert loops[0] == loops[1]


def test_flag_defaults_and_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import validate
    from src.phase_j.live_bridge import RecognitionBridge
    from src.phase_j.launcher import parse_user_config
    from pathlib import Path
    assert validate({})['live_public_match_number'] is False
    with pytest.raises(ValueError):
        validate(dict(live_public_match_number=1))
    user = parse_user_config(dict(live_public_match_number=True, device_name='OBS'), Path('.'))
    assert user.live_public_match_number
    monkeypatch.setenv('PUYO_LIVE_PUBLIC_MATCH_NUMBER', '1')
    bridge = RecognitionBridge(False, lambda *a: None)
    bridge.input_boundary_recovery = True
    bridge.record_public_boundary(1, True, False)
    bridge.record_public_boundary(2, False, True)
    assert bridge.public_matches.number(2) == 1


@pytest.mark.parametrize('device,expected', [('OBS', 0), ('capture-card', 1), (None, 1)])
def test_real_input_control_distinguishes_device_switch(device: str | None, expected: int) -> None:
    from src.phase_j.live_process import ProcessRecognitionBridge
    bridge = ProcessRecognitionBridge.__new__(ProcessRecognitionBridge)
    bridge.on_hold = lambda value: None
    bridge.public_matches = PublicMatchNumber()
    bridge._message(('hold', dict(phase='ready', epoch=0, device_id='OBS')))
    bridge._message(('hold', dict(phase='ready', epoch=1, device_id=device)))
    assert bridge.consume_input_boundary()
    assert not bridge.consume_input_boundary()
    bridge.record_public_boundary(1, True, False)
    assert bridge.public_matches.number(1) == expected


def test_error_hold_and_input_hold_keep_public_number() -> None:
    from tests.phase_j.test_live_terminal_hold import publisher, publish, terminal
    owner = publisher()
    owner.public_matches = PublicMatchNumber()
    owner.public_matches.boundary(1, True, False, True)
    publish(owner, dict(terminal(1), source='G_fe', probability=.5), 10.)
    owner.input_pending(dict(phase='verifying', progress=0, at=11.))
    assert owner.latest['game'] == 1 and owner.latest['public_game'] == 0
    owner.evaluation_hold(NS(frame=2, t_sec=12., captured_at=12., recognized_at=12.), 1)
    assert owner.latest['game'] == 1 and owner.latest['public_game'] == 0


def test_sticky_death_uses_internal_key_even_with_same_public_number() -> None:
    from src.death_confirmation import DeathConfirmTracker, DEATH_SOURCE_OJAMA
    numbers = PublicMatchNumber()
    numbers.boundary(1, True, False, True)
    tracker = DeathConfirmTracker()
    tracker._pending_source, tracker._pending_game_idx = DEATH_SOURCE_OJAMA, 0
    assert tracker.on_game_boundary(game_idx=0)[1] == 'confirmed'
    assert numbers.number(0) == numbers.number(1)
    assert tracker.resolved_is_dead_for_game(0)
    assert not tracker.resolved_is_dead_for_game(1)
    numbers.boundary(2, False, True, False)
    assert not tracker.resolved_is_dead_for_game(2)


def test_existing_terminal_recovery_carry_ends_at_score_boundary() -> None:
    from tests.phase_j.test_live_terminal_hold import publisher, publish, terminal, probability
    owner = publisher()
    owner.terminal.input_recovery = True
    owner.public_matches = PublicMatchNumber()
    publish(owner, terminal(), 10.)
    owner.public_matches.boundary(1, True, False, True)
    carried = publish(owner, dict(terminal(1), raw_probability=None,
        terminal_boundary_reason='input_recovery'), 11.)
    assert probability(carried) == 1.
    owner.public_matches.boundary(2, False, True, True)
    released = publish(owner, dict(terminal(2), raw_probability=None,
        terminal_boundary_reason='scored'), 12.)
    assert probability(released) is None
    assert released['identity']['match_id'] == '1'


def test_legacy_coalescing_cannot_be_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.visualize_advantage_overlay as module
    from src.phase_j.live_bridge import RecognitionBridge, build_live_generate
    monkeypatch.setenv('PUYO_LIVE_PUBLIC_MATCH_NUMBER', '1')
    bridge = RecognitionBridge(False, lambda *a: None)
    bridge.recovery_boundary_coalescing = True
    with pytest.raises(ValueError, match='併用'):
        build_live_generate(module, bridge)
