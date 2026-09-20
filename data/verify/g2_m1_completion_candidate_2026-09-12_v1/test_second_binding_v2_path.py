"""実loaderと同じcascade v2派生型でも基準/通常手/退役の全対照を通す。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import test_second_physical as BASE
from test_second_mode_binding import *


@pytest.fixture(scope='module')
def physical() -> Any:
    original = BASE.physical.__wrapped__()
    path = Path(original.__file__).with_name('mode_v2.py')
    spec = importlib.util.spec_from_file_location('_selected_second_v2_test', path)
    value = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('mode')
    try:
        sys.modules['mode'] = original
        spec.loader.exec_module(value)
    finally:
        if previous is None: sys.modules.pop('mode', None)
        else: sys.modules['mode'] = previous
    assert value.V1 is original
    return value
