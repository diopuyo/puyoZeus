"""設定統合、旧設定の読み替え、窓を残す停止を検証する。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.phase_j.overlay_config import validate_overlay
from src.phase_j.distribution_runtime import DistributionRuntime
from src.phase_j.distribution_window import AppWindow
from tests.phase_j.test_distribution_app import distribution, post

CORNERS = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
REMOVED = ('x', 'y', 'corner', 'mascot_flip_right', 'mascot_band_edges',
           'numbers', 'recent', 'full', 'mascot_motion', 'mascot_expressions')


@pytest.mark.parametrize('corner', CORNERS)
def test_old_settings_are_saved_without_removed_fields(corner: str) -> None:
    old = dict(layout='l', corner=corner, x=100, y=200, mascot_flip_right=True,
               mascot_band_edges=[20, 40, 60, 80], numbers=False, recent=False,
               full=True, mascot_motion=True, mascot_expressions=True)
    result = validate_overlay(old)
    assert result == dict(layout='l-' + corner, mascot_dopamine_mode=True)
    assert old['x'] == 100


@pytest.mark.parametrize('motion,expressions', [(True, True), (True, False),
                                              (False, True), (False, False)])
def test_old_dopamine_mode_uses_both_flags(motion: bool, expressions: bool) -> None:
    assert validate_overlay(dict(mascot_motion=motion, mascot_expressions=expressions)) == {
        'mascot_dopamine_mode': motion and expressions}


def test_new_dopamine_mode_wins_over_saved_legacy_flags() -> None:
    assert validate_overlay(dict(mascot_motion=False, mascot_dopamine_mode=True)) == {
        'mascot_dopamine_mode': True}


@pytest.mark.parametrize('enabled', [False, True])
def test_logs_open_in_current_window_or_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              enabled: bool) -> None:
    app = AppWindow('http://localhost:8080/setup', tmp_path, enabled)
    browser = Mock()
    monkeypatch.setattr('webbrowser.open', browser)
    app.ready.set()
    if enabled:
        app.window = Mock()
    app.open('/settings#logs')
    if enabled:
        app.window.load_url.assert_called_once_with('http://localhost:8080/settings#logs')
        app.window.show.assert_called_once()
        browser.assert_not_called()
    else:
        browser.assert_called_once_with('http://localhost:8080/settings#logs')


def test_logs_requested_before_gui_ready_keep_destination(tmp_path: Path) -> None:
    app = AppWindow('http://localhost:8080/setup', tmp_path, True)
    app.open('/settings#logs')
    app.window = Mock()
    app.shown()
    app.open()
    app.window.load_url.assert_called_once_with('http://localhost:8080/settings#logs')


def test_pause_keeps_server_and_session_marker(distribution: tuple) -> None:
    from urllib.request import urlopen
    runtime, url = distribution
    runtime.begin_session()
    assert post(url + '/api/pause', {})['stopped']
    assert (runtime.logs / 'session.running').exists()
    assert urlopen(url + '/api/state').status == 200
    runtime.stop()
    assert not (runtime.logs / 'session.running').exists()


def test_start_stop_restart_uses_new_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.phase_j.distribution_runtime as module
    runtime = DistributionRuntime(tmp_path)
    runtime.settings.update(auto_input=False, check_updates=False)
    runtime.begin_session()
    children = [Mock(), Mock()]
    for child in children:
        child.poll.return_value = None
    popen = Mock(side_effect=children)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module.subprocess, 'run', Mock())
    monkeypatch.setattr(module, 'create_job', Mock())
    monkeypatch.setattr(module, 'close_job', Mock())
    monkeypatch.setattr(module, 'assign_job', Mock())
    monkeypatch.setattr(module.threading, 'Thread', Mock())
    runtime.start()
    assert runtime.child is children[0]
    runtime.stop(final=False)
    assert runtime.child is None and runtime.stopping
    runtime.start()
    assert runtime.child is children[1] and not runtime.stopping
    assert popen.call_count == 2
    runtime.stop()


@pytest.mark.parametrize('autostart,completed,setup,expected', [
    (False, True, False, False), (True, True, False, True),
    (True, False, False, False), (True, True, True, False)])
def test_only_windows_autostart_starts_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             autostart: bool, completed: bool,
                                             setup: bool, expected: bool) -> None:
    import src.phase_j.distribution_app as app
    runtime = Mock(root=tmp_path, port=8080)
    runtime.settings = dict(port=8080, completed=completed, distribution_app_window=True, check_updates=False)
    starts = []
    monkeypatch.setattr(app, 'migrate_shortcuts', Mock())
    monkeypatch.setattr(app, 'make_server', Mock(return_value=Mock(next_server=None)))
    monkeypatch.setattr(app, 'AppWindow', Mock())
    monkeypatch.setattr(app.signal, 'signal', Mock())
    monkeypatch.setattr('src.phase_j.distribution_tray.start_tray', Mock())
    monkeypatch.setattr(app.threading, 'Thread', lambda target, **kw: Mock(
        start=lambda: starts.append(target)))
    monkeypatch.setattr(app.threading, 'Timer', Mock())
    app.run_application(runtime, SimpleNamespace(port=8080, setup=setup,
                        no_browser=False, autostart=autostart))
    assert (runtime.start_autostart in starts) is expected
    assert runtime.start not in starts


def test_autostart_shortcut_has_dedicated_argument(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import src.phase_j.distribution_windows as windows
    monkeypatch.setattr(windows, 'shortcut_directories', lambda: {
        'autostart': tmp_path / 'startup', 'desktop_shortcut': tmp_path / 'desktop'})
    run = Mock()
    monkeypatch.setattr(windows.subprocess, 'run', run)
    windows.shortcuts(tmp_path, dict(autostart=True, desktop_shortcut=True))
    assert [call.kwargs['env']['PUYO_ARGUMENTS'] for call in run.call_args_list] == [
        '-c "pass" --autostart', '']
    assert '$s.Arguments=$env:PUYO_ARGUMENTS' in run.call_args.args[0][-1]


def test_autostart_argument_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.distribution_app import parse_options
    monkeypatch.setattr('sys.argv', ['puyoZeus', '--autostart'])
    assert parse_options().autostart is True


def test_existing_autostart_shortcut_is_updated(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import src.phase_j.distribution_windows as windows
    shortcuts = Mock()
    monkeypatch.setattr(windows, 'shortcuts', shortcuts)
    windows.migrate_shortcuts(tmp_path, dict(autostart=True))
    shortcuts.assert_called_once_with(tmp_path, dict(autostart=True))


def test_old_watcher_cannot_report_failure_after_restart(tmp_path: Path) -> None:
    runtime = DistributionRuntime(tmp_path)
    previous = Mock()
    previous.stdout.readline.return_value = b''
    previous.wait.return_value = 0
    runtime.child = Mock()
    runtime._watch_child(previous)
    assert runtime.error == {}
