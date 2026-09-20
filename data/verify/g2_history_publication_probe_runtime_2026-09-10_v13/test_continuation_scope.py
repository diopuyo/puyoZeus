"""延長範囲と原認識コード不変を確認する。実tail陽性とはしない。"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Any
import common as K

PRIOR = K.VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v11'
CHANGED = frozenset(('common.py', 'launcher.sh', 'PLAN.md', 'test_continuation_scope.py',
    'session.py', 'live_cli.py', 'test_probe.py', 'REPEATED_DEPENDENCIES.json', 'repeated_connection.py',
    'derivation_check.py', 'test_scope_connection.py', 'run_cpu.py', 'closure.py',
    'finalizer_connection.py', 'test_finalizer_connection.py', 'saved_finalizer_cpu.py'))


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
    import derivation_check as D
    D.verify_common()


def test_session_change_only_repeated_dependency_scope() -> None:
    import derivation_check as D
    assert ast.dump(ast.parse((K.ROOT/'session.py').read_text())) == ast.dump(D.expected('session.py'))


def test_closure_change_only_evidence_finalizer() -> None:
    import derivation_check as D
    assert ast.dump(ast.parse((K.ROOT/'closure.py').read_text())) == ast.dump(D.expected('closure.py'))
