"""実終了関数の保存読取・索引・aliasと復元を、GPUなしで検査する。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

import pytest

from scripts import g3_postcommit_finish as F
from scripts import g3_postcommit_binding as B
from scripts import g3_postcommit_rows as R


def load(alias: str, path: Path, monkeypatch: Any) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    stack.callback(setattr, owner, name, getattr(owner, name))
    setattr(owner, name, value)


def test_world_index_same_late_library_and_restore(tmp_path: Path, monkeypatch: Any) -> None:
    fusion = load('_g3_finish_fusion_test',
                  F.ROOT / 'g2_conditional_finalizer_fusion_2026-09-10_v1/fusion.py', monkeypatch)
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    rows.append(dict(frame_idx=0, full_before={'board': [1]}, full_after={'board': [1]}))
    before, paths = dict(sys.modules), list(sys.path)
    try:
        with ExitStack() as stack:
            world = F.world_index(stack, fusion, replace)
            private = load('_g3_finish_private_world_test',
                           F.ROOT / 'g2_private_suffix_world_2026-09-10_v1/world.py', monkeypatch)
            empty = load('_g3_finish_empty_world_test',
                         F.ROOT / 'g2_empty_tail_finalizer_2026-09-10_v1/empty_world.py', monkeypatch)
            assert private.OLD == fusion.WORLD and private.libraries() is world
            with empty.loaded() as old:
                assert old.libraries() is world and old.libraries().A is world.A
            index = world.A.indexed(rows, lambda row: row['frame_idx'])
            assert isinstance(index, R.RowIndex) and index[0] == rows[0]
            rows.append(dict(frame_idx=0))
            with pytest.raises(ValueError, match='duplicate_rows'):
                world.A.indexed(rows, lambda row: row['frame_idx'])
        assert sys.path == paths and all(sys.modules.get(k) is v for k, v in before.items())
        assert 'world_verify' in before or 'world_verify' not in sys.modules
    finally:
        rows.close()


def test_original_fallback_captured_alias_and_restore(tmp_path: Path, monkeypatch: Any) -> None:
    def evaluate(*args: Any, **kwargs: Any) -> Any:
        return kwargs['outer_rows']()
    monkeypatch.setitem(sys.modules, 'fusion', N(evaluate=evaluate))
    module = load('_g3_fallback_test',
                  F.ROOT / 'g2_conditional_finalizer_fusion_2026-09-10_v1/live_connection.py', monkeypatch)
    rows = R.SavedRows(tmp_path / B.SIDECAR_NAME)
    rows.append(dict(frame_idx=0, full_before={'x': 1}, full_after={'x': 1}))
    rows.export(tmp_path / B.ROWS_NAME)
    (tmp_path / 'atomic_journal.jsonl').write_text('', encoding='utf-8')
    provider = object()
    control = N(provider=provider, calls=[], tickets=[], hidden_current_events=[],
                hidden_history_rows=[], hidden_lifetime_rows=[])
    state = dict(repeated_firing_constructor=dict(installed=True, closed=True,
                 references_restored=True, rows=[]), conditional_runtime_factory=N(controller=control, provider=provider),
                 combined_restored=lambda: None, conditional_full_installs=1,
                 conditional_revision_connection=dict(installs=1), combined_install=dict(configure_calls=1),
                 conditional_full_rows=[])
    alias, old_code = module.evaluate, module.evaluate.__code__
    captured = lambda: alias(None, [], None, tmp_path, state)
    try:
        with ExitStack() as stack:
            F.fallback(stack, module, rows, replace)
            assert captured() is rows and module.evaluate is alias
            (tmp_path / B.ROWS_NAME).write_bytes(b'[]')
            with pytest.raises(ValueError, match='changed'):
                captured()
        assert module.evaluate.__code__ is old_code and F.HELPER not in vars(module)
    finally:
        rows.close()


def test_actual_goals_multiple_scans_same_values(tmp_path: Path, monkeypatch: Any) -> None:
    def require(value: bool, reason: str) -> None:
        if not value:
            raise ValueError(reason)
    common = N(require=require, FRAMES=(0, 2), FPS=60,
               read=lambda path: json.loads(path.read_bytes()))
    monkeypatch.setitem(sys.modules, 'common', common)
    goals = load('_g3_saved_goals_test',
                 F.ROOT / 'g2_history_publication_probe_runtime_2026-09-10_v11/goals.py', monkeypatch)
    rows = R.SavedRows(tmp_path / B.SIDECAR_NAME)
    for frame in common.FRAMES:
        rows.append(dict(frame_idx=frame, time_sec=frame/60, old_projection_join_verified=True,
                         comparison_completed=True, tickets_this_update=0, released_this_update=0, changed_sides=[]))
    rows.export(tmp_path / B.ROWS_NAME)
    (tmp_path / goals.PUB_STATUS).write_text(json.dumps(dict(receiver_closed=True, receiver_errors=[],
                                                           issued=0, released=0)), encoding='utf-8')
    (tmp_path / goals.OUTER_COMPARISON).write_text(json.dumps(dict(all_consumers_ready=True, equal=True)), encoding='utf-8')
    try:
        expected = goals.publication(tmp_path, 0)
        original = common.read
        with ExitStack() as stack:
            replace(stack, common, 'read', B.reader(common.read, rows))
            assert goals.publication(tmp_path, 0) == expected
            with pytest.raises(ValueError, match='coverage'):
                common.FRAMES = (0, 2, 4)
                goals.publication(tmp_path, 0)
        assert common.read is original
    finally:
        rows.close()


def test_original_session_reclaims_world_before_outer(tmp_path: Path, monkeypatch: Any) -> None:
    fusion = load('_g3_reclaim_fusion_test',
                  F.ROOT / 'g2_conditional_finalizer_fusion_2026-09-10_v1/fusion.py', monkeypatch)
    monkeypatch.setitem(sys.modules, 'common', N(PROJECT=F.ROOT.parents[1]))
    monkeypatch.setitem(sys.modules, 'assembly_publication_probe', N())
    monkeypatch.setitem(sys.modules, 'repeated_connection', N(load=lambda stack: None))
    session = load('_g3_reclaim_original_session',
                   F.ROOT / 'g2_history_publication_probe_runtime_2026-09-10_v13/session.py', monkeypatch)
    @contextmanager
    def installed() -> Any:
        yield {}
    monkeypatch.setattr(session, 'installed', installed)
    before = set(sys.modules)
    with ExitStack() as stack:
        with session.configured():
            F.world_index(stack, fusion, replace)
            assert 'conditional_finalizer_fixed' in sys.modules
        assert 'conditional_finalizer_fixed' not in sys.modules
    assert set(sys.modules) == before
