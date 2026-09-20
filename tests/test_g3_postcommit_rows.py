"""元Consumerの正常/失敗保存と軽量索引の反例を小CPUで検査する。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import g3_postcommit_rows as R
from scripts import g3_postcommit_binding as B

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'data/verify/g2_history_publication_consumer_2026-09-09_v2/adapter.py'


def load(alias: str, path: Path, monkeypatch: Any) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    stack.callback(setattr, owner, name, getattr(owner, name))
    setattr(owner, name, value)


@pytest.mark.parametrize('failed', [False, True])
def test_original_consumer_values_and_failure(tmp_path: Path, monkeypatch: Any, failed: bool) -> None:
    module = load('adapter', ADAPTER, monkeypatch)
    fixture = load('_g3_postcommit_fixture', ADAPTER.with_name('test_adapter.py'), monkeypatch)
    baseline, candidate = tmp_path / 'baseline', tmp_path / 'candidate'
    baseline.mkdir(); candidate.mkdir()
    marker = RuntimeError('original_failure') if failed else None
    old = fixture.consumer(baseline, fixture.Comparison(failure=marker))
    new = fixture.consumer(candidate, fixture.Comparison(failure=marker))
    original_write = module.write
    with ExitStack() as stack:
        saved = B.bind(stack, new, module, replace, source=ADAPTER,
                       source_sha=B.SOURCE_SHA, verify_root=tmp_path)
        for consumer in (old, new):
            value = fixture.sample()
            if failed:
                with pytest.raises(RuntimeError) as error:
                    consumer.finish(lambda result: result, value)
                assert error.value is marker
            else:
                assert consumer.finish(lambda result: result, value) is value
            consumer.close()
        for path in baseline.iterdir():
            assert json.loads(path.read_bytes()) == json.loads((candidate / path.name).read_bytes())
        assert saved.saved(candidate / B.ROWS_NAME) is saved
    assert saved.stream.closed and module.write is original_write


def test_order_saved_changes_and_exclusive_output(tmp_path: Path) -> None:
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    try:
        for frame in (0, 2, 4):
            rows.append(dict(frame_idx=frame, full_before={'盤面': [1, 2]}))
        assert [r['frame_idx'] for r in rows] == [0, 2, 4]
        assert rows[-1]['frame_idx'] == 4 and rows[0]['frame_idx'] == 0
        rows.export(tmp_path / 'rows.json')
        assert json.loads((tmp_path / 'rows.json').read_bytes()) == list(rows)
        with pytest.raises(ValueError, match='duplicate'):
            rows.export(tmp_path / 'rows.json')
        with rows.path.open('ab') as stream:
            stream.write(b'garbage')
        with pytest.raises(ValueError, match='unregistered'):
            list(rows)
    finally:
        rows.close()


def test_mutation_corruption_and_no_silent_empty(tmp_path: Path) -> None:
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    try:
        with pytest.raises(IndexError):
            rows[-1]
        rows.append(dict(frame_idx=0, board=[1]))
        rows[0]['board'].append(9)
        with pytest.raises(ValueError, match='mutated'):
            rows[0]
        rows.latest = None
        with rows.path.open('r+b') as stream:
            stream.write(b'!')
        with pytest.raises(ValueError, match='saved_row_changed'):
            rows[0]
    finally:
        rows.close()


def test_original_duplicate_predicate_and_bounded_index(tmp_path: Path, monkeypatch: Any) -> None:
    path = ROOT / 'data/verify/g2_live_probability_context_2026-09-12_v1/probability_boundary.py'
    module = load('_g3_probability_boundary_test', path, monkeypatch)
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    try:
        for frame in (0, 2):
            rows.append(dict(frame_idx=frame, full_after=['x' * 100_000]))
        index = R.indexed(module.indexed, rows, 'frame_idx', field=True)
        assert isinstance(index, R.RowIndex) and index[2] == rows[1]
        assert list(index) == [0, 2] and index.offsets == {0: (0, 0), 2: (2, 1)}
        rows.append(dict(frame_idx=2))
        with pytest.raises(AssertionError, match='duplicate_frame'):
            R.indexed(module.indexed, rows, 'frame_idx', field=True)
        ordinary = [dict(frame_idx=7)]
        assert R.indexed(module.indexed, ordinary, 'frame_idx', field=True)[7] is ordinary[0]
    finally:
        rows.close()


def test_append_failure_preserves_complete_prefix(tmp_path: Path, monkeypatch: Any) -> None:
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    rows.append(dict(frame_idx=0))
    monkeypatch.setattr(R, 'MAX_ROW_BYTES', 5)
    try:
        with pytest.raises(ValueError, match='too_large'):
            rows.append(dict(frame_idx=2))
        assert len(rows) == 1 and rows.failure
        with pytest.raises(ValueError, match='failed'):
            rows.export(tmp_path / 'rows.json')
    finally:
        rows.close()
    assert json.loads(rows.path.read_bytes()) == dict(frame_idx=0)


def test_real_row_reader_and_export_failure(tmp_path: Path, monkeypatch: Any) -> None:
    source = Path('/mnt/d/puyo_analyzer/verify/g3_repair_2026-09-15_v1/postcommit_storage_cpu_input/ROW.json')
    if not source.exists():
        source = Path('D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/postcommit_storage_cpu_input/ROW.json')
    row = json.loads(source.read_bytes())  # 保存結合のfixture。GTや新run真値ではない。
    rows = R.SavedRows(tmp_path / B.SIDECAR_NAME)
    target = tmp_path / B.ROWS_NAME
    try:
        rows.append(row)
        assert rows[0] == row
        read = B.reader(lambda path: json.loads(path.read_bytes()), rows)
        with pytest.raises(ValueError, match='missing'):
            read(target)
        rows.export(target)
        assert read(target) is rows and read(target)[0] == row
        other = tmp_path / 'other.json'
        other.write_text('{"ordinary":true}', encoding='utf-8')
        assert read(other) == {'ordinary': True}
        target.write_bytes(b'[]')
        with pytest.raises(ValueError, match='changed'):
            read(target)
    finally:
        rows.close()


def test_partial_write_is_sticky_and_reclaimed(tmp_path: Path) -> None:
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    rows.append(dict(frame_idx=0))
    original = rows.stream
    class Broken:
        closed = False
        def tell(self) -> int:
            return original.tell()
        def write(self, raw: bytes) -> int:
            return original.write(raw[:3])
        def flush(self) -> None:
            original.flush()
        def fileno(self) -> int:
            return original.fileno()
        def close(self) -> None:
            original.close()
            self.closed = True
    rows.stream = Broken()
    try:
        with pytest.raises(ValueError, match='short_write'):
            rows.append(dict(frame_idx=2))
        assert len(rows) == 1 and rows.failure
        with pytest.raises(ValueError, match='failed'):
            rows.append(dict(frame_idx=4))
    finally:
        rows.close()
    assert rows.stream.closed and original.closed
    assert json.loads(rows.path.read_bytes().splitlines()[0]) == dict(frame_idx=0)


def test_cleanup_failure_does_not_replace_original(tmp_path: Path, monkeypatch: Any) -> None:
    rows = R.SavedRows(tmp_path / 'rows.jsonl')
    marker = RuntimeError('primary_failure')
    secondary = OSError('close_failure')
    close = rows.close
    def broken_close() -> None:
        close()
        raise secondary
    monkeypatch.setattr(rows, 'close', broken_close)
    B.cleanup(rows, marker)
    receipt = json.loads((tmp_path / 'G3_POSTCOMMIT_END.json').read_bytes())
    assert receipt['stream_closed'] and 'primary_failure' in receipt['original_error']
    assert 'close_failure' in receipt['failure']
