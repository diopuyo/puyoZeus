"""専用writerは全行を同じbytes・順序で保存し、I/O失敗を隠さない。"""
from pathlib import Path
from threading import Event
import pytest
from src.phase_j.live_obs_audit import AsyncAudit


@pytest.mark.parametrize('limit', [64, 1024*1024])
def test_fifo_buffer_preserves_values_positions_and_file_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    """既読prefix・clear・spill・終了時の未読分まで同期版と一致する。"""
    from src.phase_j import live_spool
    monkeypatch.setattr(live_spool, 'FIFO_BUFFER_BYTES', limit)
    outputs, values = [], []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        path = tmp_path/f'fifo-{enabled}.pickle'
        fifo = live_spool.DiskFIFO(path)
        popped = []
        for index in range(12):
            row = dict(index=index, cells=[index]*8)
            fifo.append(row)
            row['cells'][0] = -1
            assert fifo.buffer is None or fifo.buffer.tell() <= limit
            if index % 3 == 0:
                popped.append(fifo.popleft())
        assert len(fifo) == 8
        fifo.clear()
        assert len(fifo) == 0
        for index in range(3):
            fifo.append(index)
            popped.append(fifo.popleft())
            assert len(fifo) == 0
        with pytest.raises(IndexError):
            fifo.popleft()
        for index in range(4):
            fifo.append([index]*8)
        popped.append(fifo.popleft())
        fifo.close()
        outputs.append(path.read_bytes())
        values.append(popped)
    assert outputs[0] == outputs[1]
    assert values[0] == values[1]


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_explicit_default_buffer_is_preserved(monkeypatch: pytest.MonkeyPatch, enabled: str) -> None:
    from io import StringIO
    from types import SimpleNamespace
    from src.phase_j.live_obs_audit import open_audit, BUFFER_BYTES
    options = []

    def opened(mode: str, **kwargs: object) -> StringIO:
        options.append(dict(mode=mode, **kwargs))
        return StringIO()

    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
    stream = open_audit(SimpleNamespace(open=opened), buffering=-1)
    stream.write('記録\n')
    stream.close()
    assert options == [dict(mode='w', encoding='utf-8',
                            buffering=BUFFER_BYTES if enabled == '1' else -1)]


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_open_audit_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: str) -> None:
    from src.phase_j.live_obs_audit import open_audit
    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
    stream = open_audit(tmp_path/'rows', 'x')
    assert isinstance(stream, AsyncAudit) == (enabled == '1')
    stream.write('観測\n')
    stream.close()
    assert (tmp_path/'rows').read_text(encoding='utf-8') == '観測\n'


def test_exact_order_and_close(tmp_path: Path) -> None:
    path = tmp_path/'audit.jsonl'
    rows = [f'{{"i":{i},"v":"観測"}}\n' for i in range(1000)]
    stream = AsyncAudit(path.open('w', encoding='utf-8', newline=''))
    for row in rows:
        stream.write(row)
    stream.close()
    stream.close()
    assert path.read_bytes() == ''.join(rows).encode('utf-8')
    assert not stream.thread.is_alive()


def test_failure_is_reported() -> None:
    failed = Event()
    class Broken:
        def write(self, value: str) -> None:
            failed.set()
            raise OSError('disk full')
        def close(self) -> None:
            pass
    stream = AsyncAudit(Broken())
    stream.write('first\n')
    assert failed.wait(2)
    stream.thread.join(2)
    with pytest.raises(OSError):
        stream.write('second\n')
    with pytest.raises(OSError):
        stream.close()


def test_binary_checkpoint_does_not_wait_for_writer() -> None:
    """位置は投入済みbytesから決まり、失敗時のseekはFIFO順で完了を待つ。"""
    from io import BytesIO
    from src.phase_j.live_obs_audit import AsyncSpool
    entered, release, finished = Event(), Event(), Event()
    class Blocked(BytesIO):
        def write(self, value: bytes) -> int:
            entered.set()
            if not release.wait(5):
                raise OSError('テストのwriter解除待ち')
            count = super().write(value)
            finished.set()
            return count
    raw = Blocked()
    stream = AsyncSpool(raw)
    try:
        stream.write(b'abc')
        assert entered.wait(2)
        assert stream.tell() == 3
        assert not finished.is_set()
        release.set()
        assert stream.seek(1) == stream.tell() == 1
        stream.write(b'de')
        assert stream.tell() == 3
        stream.flush()
        assert raw.getvalue() == b'ade'
    finally:
        release.set()
        stream.close()


def test_live_flush_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import live_obs_audit as audit
    monkeypatch.setattr(audit, 'FLUSH_SEC', .01)
    flushed, values = Event(), []
    class Stream:
        def write(self, value: str) -> None:
            values.append(value)
        def flush(self) -> None:
            flushed.set()
        def close(self) -> None:
            pass
    stream = AsyncAudit(Stream())
    try:
        stream.write('one\n')
        assert flushed.wait(2)
        assert ''.join(values) == 'one\n'
        assert stream.thread.is_alive()
    finally:
        stream.close()


def test_spool_mutation_read_clear_and_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_spool import DiskRows
    saved = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        path = tmp_path/enabled
        rows = DiskRows(path)
        for index in range(1000):
            item = dict(index=index, payload=b'x'*200)
            rows.append(item)
            item['after_append'] = index*2
        assert len(rows) == 1000
        assert rows[-1]['after_append'] == 1998
        assert [r['after_append'] for r in rows] == list(range(0, 2000, 2))
        before_clear = path.read_bytes()
        rows.clear()
        assert len(rows) == 0 and list(rows) == []
        rows.extend([dict(index=1), dict(index=2)])
        assert rows[0] == dict(index=1)
        rows.close()
        saved.append((before_clear, path.read_bytes()))
    assert saved[0] == saved[1]


def test_operation_failure_is_reported() -> None:
    class Broken:
        def flush(self) -> None:
            raise OSError('flush failed')
        def close(self) -> None:
            pass
    stream = AsyncAudit(Broken())
    with pytest.raises(OSError):
        stream.flush()
    stream.thread.join(2)
    assert not stream.thread.is_alive()
    with pytest.raises(OSError):
        stream.close()


@pytest.mark.parametrize('enabled', ['0', '1'])
def test_history_commit_rollback_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: str,
) -> None:
    """実際の履歴確定契約で、成功・途中失敗・再試行の出力を照合する。"""
    from src.phase_j.live_eval_recovery import rollback_rows
    from src.phase_j.live_spool import DiskRows
    monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
    archive, diagnostics = (DiskRows(tmp_path/name) for name in ('archive', 'diagnostics'))
    try:
        for rows in (archive, diagnostics):
            rows.extend([dict(index=0), dict(index=1)])
        with pytest.raises(RuntimeError, match='途中失敗'):
            with rollback_rows(archive, diagnostics):
                archive.extend([dict(index=2), dict(index=3)])
                diagnostics.append(dict(index=2))
                raise RuntimeError('途中失敗')
        for rows in (archive, diagnostics):
            assert list(rows) == [dict(index=0), dict(index=1)]
        with rollback_rows(archive, diagnostics):
            archive.append(dict(index=2))
            diagnostics.append(dict(index=2))
        for rows in (archive, diagnostics):
            assert list(rows) == [dict(index=i) for i in range(3)]
    finally:
        archive.close()
        diagnostics.close()
    assert (tmp_path/'archive').read_bytes() == (tmp_path/'diagnostics').read_bytes()


@pytest.mark.parametrize('compact', [False, True])
def test_ama_input_copy_only_removed_from_transport(monkeypatch: pytest.MonkeyPatch, compact: bool) -> None:
    from io import StringIO
    import json
    from queue import Queue
    from src.phase_j import live_ama_async as worker
    monkeypatch.setattr(worker, 'perf_counter', lambda: 123.)
    job = dict(item=dict(boards=[[1, 2]]), result=dict(value=dict(probability=.3)), frame=4)
    done, audit = Queue(), StringIO()
    done.put(job)
    reply = worker.collect_results(done, audit, compact)[0]
    assert json.loads(audit.getvalue()) == dict(job, collected_at=123.)
    assert reply == {k: v for k, v in job.items() if not (compact and k == 'item')}


def test_sent_map_preserves_first_value_order_and_final_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLiteの物理transaction回数によらず、論理行・重複排除・末尾保存が同じ。"""
    import pickle
    import sqlite3
    from src.phase_j.live_spool import DiskMap
    observed = []
    for enabled in ('0', '1'):
        monkeypatch.setenv('PUYO_OBS_FAST_AUDIT', enabled)
        path = tmp_path/f'sent_{enabled}.sqlite'
        rows = DiskMap(path)
        try:
            for key in (3, 1, 2, 1):
                value = dict(key=key, values=[key])
                rows.setdefault(key, value)
                value['values'].append(-1)
            assert len(rows) == 3
            assert list(rows.values()) == [dict(key=i, values=[i]) for i in (1, 2, 3)]
            rows.setdefault(4, dict(key=4, values=[4]))
        finally:
            rows.close()
        with sqlite3.connect(path) as saved:
            observed.append([(key, pickle.loads(raw)) for key, raw in
                             saved.execute('SELECT key, value FROM rows ORDER BY key')])
    assert observed[0] == observed[1] == [(i, dict(key=i, values=[i])) for i in (1, 2, 3, 4)]
