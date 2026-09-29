"""独立プロセス専用の凍結loaderを使うテストの状態を隔離する。"""
from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import Iterator

import pytest

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


@pytest.fixture(scope='module', autouse=True)
def isolate_frozen_loader(request: pytest.FixtureRequest) -> Iterator[None]:
    """ハッシュ固定された検収fixture自体を変えず、その外側を隔離する。"""
    if request.module.__name__ in FROZEN_MODULES:
        with isolated_frozen_imports():
            yield
    else:
        yield
