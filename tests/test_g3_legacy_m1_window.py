"""M1終端要求の限定接続を検査する。記録が残ることと、他の検査を緩めていないことを示す。"""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import importlib.util
import json
import sys

import pytest

from scripts import g3_legacy_m1_window as P


@dataclass(frozen=True)
class State:
    last: int | None = None
    accepted: tuple = ()
    pending: int | None = None


def loaded(monkeypatch: Any, name: str) -> Any:
    path = P.G.ROOT / P.TARGET
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def schedule(end: int = 36298) -> Any:
    def check(state: Any) -> None:
        return None

    def finish(state: Any) -> tuple:
        if len(state.accepted) < 2:
            raise ValueError('capture_schedule:incomplete_coverage')
        return tuple(state.accepted)
    return N(END=end, EARLIEST=(35370, 35410), check=check, finish=finish)


def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
    stack.callback(setattr, owner, name, getattr(owner, name))
    setattr(owner, name, value)


def test_original_rejects_unfinished_window(monkeypatch: Any, tmp_path: Path) -> None:
    """元の実装は終端未到達を observation_incomplete で拒否する。反例を先に示す。"""
    module = loaded(monkeypatch, '_m1_original')
    with pytest.raises(ValueError, match='legacy_m1_compatibility:observation_incomplete'):
        module.finished(schedule(), State(last=30000), tmp_path)


def test_patched_records_and_falls_through(monkeypatch: Any, tmp_path: Path) -> None:
    """差し替え後は止めずに記録し、原unsupported経路へ流す。"""
    module = loaded(monkeypatch, '_m1_patched')
    with ExitStack() as stack:
        receipt = P.install(stack, replace)
        assert receipt['receipts'] and receipt['receipts'][0]['file'] == P.TARGET
        module.finished(schedule(), State(last=30000), tmp_path)
        summary = receipt['summarize']()
    assert summary['call_count'] == 1
    assert summary['unsatisfied_calls'][0]['last'] == 30000
    assert summary['unsatisfied_calls'][0]['end'] == 36298
    rows = [json.loads(x) for x in (tmp_path / P.LOG).read_text(encoding='utf-8').splitlines() if x]
    assert len(rows) == 1 and rows[0]['original_require_satisfied'] is False


def test_patched_keeps_satisfied_calls_identical(monkeypatch: Any, tmp_path: Path) -> None:
    """条件を満たす呼出は記録だけ増え、判定結果は元と同じ unsupported になる。"""
    module = loaded(monkeypatch, '_m1_satisfied')
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    original = module.finished(schedule(), State(last=36298), first)
    with ExitStack() as stack:
        receipt = P.install(stack, replace)
        patched = module.finished(schedule(), State(last=36298), second)
        summary = receipt['summarize']()
    assert patched == original
    assert summary['call_count'] == 1 and summary['unsatisfied_calls'] == []


def test_patched_still_raises_unexpected_coverage(monkeypatch: Any, tmp_path: Path) -> None:
    """被覆が想定外に足りている場合は、元どおり unexpected_coverage_failure で落ちる。"""
    module = loaded(monkeypatch, '_m1_coverage')
    broken = schedule()
    broken.finish = lambda state: (_ for _ in ()).throw(ValueError('capture_schedule:incomplete_coverage'))
    with ExitStack() as stack:
        P.install(stack, replace)
        with pytest.raises(ValueError, match='unexpected_coverage_failure'):
            module.finished(broken, State(last=36298, accepted=(1, 2)), tmp_path)


def test_source_sha_guard(monkeypatch: Any) -> None:
    """原G2ファイルが変わったら黙って通さず、名前付きで落とす。"""
    loaded(monkeypatch, '_m1_sha')
    monkeypatch.setattr(P, 'TARGET_SHA', '0' * 64)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='m1_window_source_sha'):
            P.install(stack, replace)


def test_restored_after_scope(monkeypatch: Any, tmp_path: Path) -> None:
    """scope終了で元関数へ戻る。"""
    module = loaded(monkeypatch, '_m1_restore')
    original = module.finished
    with ExitStack() as stack:
        P.install(stack, replace)
        assert module.finished is not original
    assert module.finished is original


def test_summary_is_json_serializable(monkeypatch: Any, tmp_path: Path) -> None:
    """受領票はJSON化できる。v9は summarize 関数を含めたまま保存して落ちた。"""
    module = loaded(monkeypatch, '_m1_json')
    with ExitStack() as stack:
        receipt = P.install(stack, replace)
        module.finished(schedule(), State(last=30000), tmp_path)
        summary = receipt['summarize']()
    assert 'summarize' not in summary
    json.dumps(summary, ensure_ascii=False, allow_nan=False)
