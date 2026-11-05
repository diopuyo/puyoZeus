"""独立プロセス専用の凍結loaderを使うテストの状態を隔離する。"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolated_shortcut_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """設定保存の試験も、本物のデスクトップ・スタートメニューに書かせない。"""
    from src.phase_j import distribution_windows
    directories = {key: tmp_path / 'shortcuts' / key
                   for key in ('desktop_shortcut', 'start_menu_shortcut', 'autostart')}
    monkeypatch.setattr(distribution_windows, 'shortcut_directories', lambda: directories)


@pytest.fixture(autouse=True)
def site_feedback_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.6.2 で出荷の既定は ON。試験は OFF から始め、ON の試験だけが明示して入れる (本番のサイトへ送らない)。"""
    from src import production_config
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', False)


def install_site_send_guard() -> None:
    """どの試験からも本番の配布サイトへ通信させない (旗を ON にする試験の送り先の差し替え漏れも止める)。"""
    from urllib.parse import urlsplit
    from src.phase_j.distribution_config import DISTRIBUTION_SITE_URL
    site_host = urlsplit(DISTRIBUTION_SITE_URL).hostname

    def audit(event: str, args: tuple) -> None:
        if event == 'urllib.Request' and urlsplit(str(args[0])).hostname == site_host:
            raise AssertionError(f'試験から本番の配布サイトへの通信は禁止: {args[0]}')
    sys.addaudithook(audit)


install_site_send_guard()


def install_shortcut_write_guard() -> None:
    """直接のファイル操作とPowerShellへのリンクパスの引き渡しを監視する。"""
    if os.name != 'nt':
        return
    from src.phase_j.distribution_windows import shortcut_directories
    protected = tuple(path.resolve() for path in shortcut_directories().values())

    def check_path(value: object) -> None:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).resolve()
        if path.suffix.lower() == '.lnk' and any(path.is_relative_to(root) for root in protected):
            raise AssertionError(f'本物のショートカットへの書き込みは禁止: {path}')

    def audit(event: str, args: tuple) -> None:
        if event == 'open' and (args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            check_path(args[0])
        elif event in ('os.remove', 'os.rename'):
            check_path(args[0])
            if event == 'os.rename':
                check_path(args[1])
        elif event == 'subprocess.Popen':
            env = args[3] or os.environ
            check_path(env.get('PUYO_LINK'))
            command = str(args[1]).lower()
            if '.lnk' in command and any(str(root).lower() in command for root in protected):
                raise AssertionError('本物のショートカットを操作する子プロセスは禁止')
    sys.addaudithook(audit)


install_shortcut_write_guard()

FROZEN_MODULES = frozenset({
    'tests.test_diagnose_video38_next_enqueue_live_shadow_v1',
    'tests.test_next_enqueue_live_shadow_v1',
})


def frozen_module(name: str) -> bool:
    """本番srcと固定collectorのimportを一組として退避する。"""
    return name == 'src' or name.startswith('src.') or name == '_video38_frozen_collector'


@contextmanager
def isolated_frozen_imports() -> Iterator[None]:
    """setup中の例外も含め、cwd・検索順・モジュール実体を必ず復元する。"""
    saved = {name: value for name, value in sys.modules.items() if frozen_module(name)}
    paths, directory = list(sys.path), os.getcwd()
    try:
        for name in saved:
            sys.modules.pop(name, None)
        yield
    finally:
        os.chdir(directory)
        sys.path[:] = paths
        for name in list(sys.modules):
            if frozen_module(name):
                sys.modules.pop(name)
        sys.modules.update(saved)


@contextmanager
def restored_process_state() -> Iterator[None]:
    """一般moduleが変えたcwdと検索順を、次のmoduleへ持ち越さない。

    凍結snapshotへのchdirが残ると、spawn子が本番srcでなく旧srcを読み、
    cwd相対のテストも別の場所を見る。src実体は触らない（class同一性を保つ）。
    """
    paths, directory = list(sys.path), os.getcwd()
    try:
        yield
    finally:
        os.chdir(directory)
        sys.path[:] = paths


@pytest.fixture(scope='module', autouse=True)
def isolate_frozen_loader(request: pytest.FixtureRequest) -> Iterator[None]:
    """ハッシュ固定された検収fixture自体を変えず、その外側を隔離する。"""
    if request.module.__name__ in FROZEN_MODULES:
        with isolated_frozen_imports():
            yield
    else:
        with restored_process_state():
            yield
