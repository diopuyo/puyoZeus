"""延長範囲と原認識コード不変を確認する。実tail陽性とはしない。"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any
import common as K

PRIOR = K.VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v9'
CHANGED = frozenset(('common.py', 'run_cpu.py', 'launcher.sh', 'PLAN.md', 'test_continuation_scope.py', 'test_missing_connection.py', 'missing_connection.py', 'assembly_publication_probe.py'))


def test_same_recognition_sources() -> None:
    for name in set(K.OWN)-CHANGED:
        assert (K.ROOT/name).read_bytes() == (PRIOR/name).read_bytes(), name


def test_original_start_and_extended_end() -> None:
    assert K.FIRST == 29052 and K.END == 36300 and K.STRIDE == 2 and K.FPS == 60
    assert len(K.FRAMES) == 3624 and K.FRAMES[-1] == 36298
    assert K.bounds()['original_end_sec'] == K.END/K.FPS == 605.0
    assert K.HISTORY_FIRST == 34796


def test_kwargs_only_interval() -> None:
    original = dict(enable_ojama_write_accounting_guard=True, normalize_fps_30=True,
        sample_interval_sec=0, user_setting=object(), start_sec=484.2, max_sec=120.8)
    actual = K.actual_kwargs(original)
    assert actual['user_setting'] is original['user_setting']
    assert actual['start_sec'] == 484.2 and actual['max_sec'] == 120.8
    assert actual['precise_seek'] is False and original.get('precise_seek') is None


def test_common_change_only_missing_connection() -> None:
    old = (PRIOR/'common.py').read_text()
    insertion = "MISSING = VERIFY / 'g2_history_missing_observation_2026-09-10_v1'\n"
    insertion += "FIXED[MISSING / 'connection.py'] = 'cb5f501efafc91c1ebc47accea739b7d47226ad10487a1c0db2a2a2c7dcc46a0'\n\n"
    expected = old.replace("OWN = (", insertion + "OWN = ('missing_connection.py', 'test_missing_connection.py', ")
    assert (K.ROOT/'common.py').read_text() == expected
