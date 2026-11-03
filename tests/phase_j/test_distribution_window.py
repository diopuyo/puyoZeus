"""窓の開閉・既存インスタンス・依存不在時のブラウザ退避を検証する。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
from typing import Any, Callable
from unittest.mock import Mock

import pytest

from src.phase_j.distribution_config import validate_settings
from src.phase_j.distribution_window import AppWindow, request_existing


def test_flag_defaults_on_and_accepts_off() -> None:
    assert validate_settings({})['distribution_app_window'] is True
    assert not validate_settings({'distribution_app_window': False})['distribution_app_window']
    with pytest.raises(ValueError):
        validate_settings({'distribution_app_window': 1})


def test_close_hides_without_finishing(tmp_path: Path) -> None:
    app = AppWindow('http://localhost/setup', tmp_path, True)
    app.window = Mock()
    assert app.closing() is False
    app.window.hide.assert_called_once()
    assert not app.finished.is_set()
    app.exiting = True
    assert app.closing() is True


def test_open_restores_and_queues_before_ready(tmp_path: Path) -> None:
    app = AppWindow('http://localhost/setup', tmp_path, True)
    app.window = Mock()
    app.open()
    assert app.pending_open.is_set()
    app.window.show.assert_not_called()
    app.shown()
    app.open()
    app.window.show.assert_called_once()
    app.window.restore.assert_called_once()


@pytest.mark.parametrize('backend', [None, 'mshtml'])
def test_missing_dependency_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     backend: str | None) -> None:
    app = AppWindow('http://localhost/setup', tmp_path, True)
    module = None if backend is None else SimpleNamespace(
        initialize=lambda gui: SimpleNamespace(renderer=backend))
    monkeypatch.setitem(sys.modules, 'webview', module)
    browser = Mock(side_effect=lambda url: app.finished.set())
    monkeypatch.setattr('webbrowser.open', browser)
    app.run()
    browser.assert_called_once_with(app.url)
    assert app.window is None
    assert app.enabled is False


def test_duplicate_instance_opens_and_exit_destroys(tmp_path: Path) -> None:
    (tmp_path / 'logs').mkdir()
    app = AppWindow('http://localhost/setup', tmp_path, True)
    opened = threading.Event()
    app.window = Mock()
    app.window.restore.side_effect = opened.set
    app.shown()
    worker = threading.Thread(target=app.watch)
    worker.start()
    try:
        request_existing(tmp_path)
        assert opened.wait(3)
    finally:
        app.finished.set()
        worker.join(timeout=3)
    app.window.destroy.assert_called_once()


def test_no_browser_suppresses_initial_open(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    app = AppWindow('http://localhost/setup', tmp_path, False, hidden=True)
    browser = Mock()
    monkeypatch.setattr('webbrowser.open', browser)
    app.finished.set()
    app.run()
    browser.assert_not_called()


class WindowEvent:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], object]] = []

    def __iadd__(self, callback: Callable[[], object]) -> 'WindowEvent':
        self.callbacks.append(callback)
        return self

    def emit(self) -> None:
        for callback in self.callbacks:
            callback()


@pytest.mark.parametrize('hidden', [False, True])
def test_enabled_gui_uses_main_loop_and_explicit_initial_show(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden: bool) -> None:
    app = AppWindow('http://localhost/setup', tmp_path, True, hidden=hidden)
    destroyed = threading.Event()
    window = Mock()
    window.events = SimpleNamespace(closing=WindowEvent(), shown=WindowEvent(),
                                   before_show=WindowEvent())
    window.destroy.side_effect = destroyed.set
    def start(**kwargs: Any) -> None:
        window.events.shown.emit()
        assert kwargs['gui'] == 'edgechromium'
        assert (kwargs['func'] is None) == hidden
        if kwargs['func'] is not None:
            kwargs['func']()
        app.finished.set()
    module = SimpleNamespace(initialize=Mock(return_value=SimpleNamespace(renderer='edgechromium')),
                             create_window=Mock(return_value=window), start=start)
    monkeypatch.setitem(sys.modules, 'webview', module)
    app.run()
    assert destroyed.wait(3)
    assert module.create_window.call_args.kwargs['hidden'] is hidden
    assert window.show.call_count == (0 if hidden else 1)
    assert window.restore.call_count == (0 if hidden else 1)
