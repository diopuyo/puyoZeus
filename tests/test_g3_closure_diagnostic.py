"""sink複合検査の分解記録を検査する。どの部分条件が偽かを残せることを示す。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import json

import pytest

from scripts import g3_closure_diagnostic as P


def sink(closed: bool = True, errors: Any = ()) -> Any:
    return N(closed=closed, errors=errors, path='/tmp/sink.jsonl')


def handle(closed: bool) -> Any:
    return N(closed=closed)


def state(tmp_path: Path) -> dict:
    return {'output': str(tmp_path)}


def observe(record: list) -> Any:
    return P.observer(record)


class FakeRows:
    """SavedRows と同じ形。close は多重呼出しに耐える。"""

    def __init__(self, path: str, closed: bool = False) -> None:
        self.stream = N(name=path, closed=closed)
        self.closes = 0

    def close(self) -> None:
        self.closes += 1
        self.stream.closed = True


def test_sidecar_closed_before_check(tmp_path: Path) -> None:
    """検査の直前に副次streamを閉じる。v11で唯一未閉鎖だったもの。"""
    rows = FakeRows('/x/' + P.SIDECAR)
    consumer = N(rows=rows)
    st = {'output': str(tmp_path), 'postcommit_publication_consumer': consumer}
    record: list = []
    handle = N(closed=False)
    closed = P.close_sidecar(st)
    assert len(closed) == 1 and closed[0]['was_closed'] is False and closed[0]['now_closed'] is True
    assert rows.closes == 1 and rows.stream.closed is True


def test_sidecar_close_is_idempotent(tmp_path: Path) -> None:
    """既に閉じていても壊れない。後段の原closeが残っても二重で問題にならない。"""
    rows = FakeRows('/x/' + P.SIDECAR, closed=True)
    st = {'output': str(tmp_path), 'c': N(rows=rows)}
    closed = P.close_sidecar(st)
    assert closed[0]['was_closed'] is True and closed[0]['error'] is None


def test_sidecar_only_targets_the_named_file(tmp_path: Path) -> None:
    """他のstreamは閉じない。対象は副次streamの1本だけ。"""
    other = FakeRows('/x/frames.jsonl')
    st = {'output': str(tmp_path), 'c': N(rows=other)}
    assert P.close_sidecar(st) == []
    assert other.closes == 0


def test_records_all_four_subconditions(tmp_path: Path) -> None:
    """4条件を個別に残す。ANDの結果だけにしない。"""
    record: list = []
    observe(record)(sink(), {'a': handle(True), 'b': handle(True)}, state(tmp_path))
    item = record[0]
    assert item['sink_closed'] is True
    assert item['sink_errors_empty'] is True
    assert item['handles_present'] is True
    assert item['handles_all_closed'] is True
    assert item['original_require_satisfied'] is True
    assert item['unclosed_handles'] == []


def test_identifies_the_unclosed_handle(tmp_path: Path) -> None:
    """閉じていないstreamの名前を残す。v10はこれが分からず原因不明だった。"""
    record: list = []
    with pytest.raises(ValueError, match='sink_stream_unclosed_diagnosed'):
        observe(record)(sink(), {'ok': handle(True), 'stuck': handle(False)}, state(tmp_path))
    item = record[0]
    assert item['original_require_satisfied'] is False
    assert item['unclosed_handles'] == ['stuck']
    assert item['handles_all_closed'] is False
    assert item['sink_closed'] is True


def test_distinguishes_sink_from_handles(tmp_path: Path) -> None:
    """sink側が原因の場合と handles 側が原因の場合を区別できる。"""
    record: list = []
    with pytest.raises(ValueError, match='sink_stream_unclosed_diagnosed'):
        observe(record)(sink(closed=False), {'a': handle(True)}, state(tmp_path))
    assert record[0]['sink_closed'] is False and record[0]['handles_all_closed'] is True


def test_empty_handles_is_not_success(tmp_path: Path) -> None:
    """handlesが空のときも成功にしない。母数0を合格と読み替えない。"""
    record: list = []
    with pytest.raises(ValueError, match='sink_stream_unclosed_diagnosed'):
        observe(record)(sink(), {}, state(tmp_path))
    assert record[0]['handles_present'] is False
    assert record[0]['original_require_satisfied'] is False


def test_report_written_and_serializable(tmp_path: Path) -> None:
    """原票が書かれ、JSON化できる。v9は関数を混ぜて保存に失敗した。"""
    record: list = []
    with pytest.raises(ValueError, match='sink_stream_unclosed_diagnosed'):
        observe(record)(sink(), {'a': handle(False)}, state(tmp_path))
    path = tmp_path / P.REPORT
    assert path.exists()
    value = json.loads(path.read_text(encoding='utf-8'))
    assert value['calls'][0]['unclosed_handles'] == ['a']
    json.dumps(value, ensure_ascii=False, allow_nan=False)


def test_not_tolerated_raises_named(tmp_path: Path) -> None:
    """TOLERATE=False では未閉鎖を名前つきで拒否する。黙って通さない。"""
    assert P.TOLERATE is False
    record: list = []
    with pytest.raises(ValueError, match='sink_stream_unclosed_diagnosed'):
        P.observer(record)(sink(), {'a': handle(False)}, state(tmp_path))
    assert record[0]['unclosed_handles'] == ['a']


def test_source_sha_guard(monkeypatch: Any) -> None:
    """原G2ファイルが変われば名前つきで落とす。"""
    monkeypatch.setattr(P, 'TARGET_SHA', '0' * 64)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='closure_diagnostic_source_sha'):
            P.rebuilt(N(__file__=str(P.G.ROOT / P.TARGET)), [])


def test_rebuilt_replaces_exactly_one_line() -> None:
    """差し替え対象の行が原ファイルに1つだけ存在する。"""
    text = (P.G.ROOT / P.TARGET).read_text(encoding='utf-8')
    assert text.count(P.OLD_LINE) == 1
