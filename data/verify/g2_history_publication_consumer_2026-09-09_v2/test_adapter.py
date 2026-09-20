"""比較hookの順序/例外/非変異境界。collector helper自体は全結合で別検収。"""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any
import pytest
import adapter as A


@dataclass
class Result:
    frame_idx: int
    time_sec: float
    p1: Any
    p2: Any
    other: Any


class Comparison:
    def __init__(self, mutation: bool = False, failure: Any = None) -> None:
        self.calls: list[Any] = []
        self.mutation, self.failure = mutation, failure
    def consume(self, before: Any, after: Any, lock: Any, tsumo: Any) -> None:
        self.calls.append((before,after,lock,tsumo))
        if self.mutation:
            after.p1['value'] = 'wrong'
        if self.failure:
            raise self.failure
    def snapshot(self) -> Any:
        return dict(calls=len(self.calls),quality_gate_clear=False)


def consumer(path: Path, comparison: Any) -> Any:
    pipe = SimpleNamespace(_post_match_lockdown_active=False,tsumo_count=lambda side: 10)
    entry = dict(frame_idx=100,time_sec=100/60,returned={'1P':dict(value='held'),'2P':dict(value='2p')},
                 actual_tsumo_getter=[10,10])
    receiver = SimpleNamespace(active=dict(pipe=pipe,tickets={}),registration={},issued=0,released=0,
        consumer=SimpleNamespace(publisher=SimpleNamespace(entries=[entry])),
        rec=SimpleNamespace(side_value=lambda side: dict(side),expected={100}))
    return A.Consumer(receiver,comparison,path)


def sample() -> Result:
    return Result(100,100/60,dict(value='held'),dict(value='2p'),{'mutable':[]})


def test_comparison_receives_new_outer_result(tmp_path: Path) -> None:
    comparison = Comparison()
    value, original = consumer(tmp_path,comparison), sample()
    final = replace(original,p1=dict(value='released'))
    assert value.finish(lambda _:final,original) is final
    assert comparison.calls[0][:2] == (original,final)
    assert value.rows[0]['changed_sides'] == ['1P'] and original.p1['value']=='held'
    assert not value.rows[0]['actual_collector_append_verified']
    value.close()
    assert (tmp_path/A.COMPARISON).is_file() and value.closed


@pytest.mark.parametrize('mutation',[False,True])
def test_comparison_failure_returns_no_result(tmp_path: Path, mutation: bool) -> None:
    marker = RuntimeError('comparison_failure')
    value = consumer(tmp_path,Comparison(mutation=mutation,failure=None if mutation else marker))
    original = sample()
    with pytest.raises((RuntimeError,ValueError)) as error:
        value.finish(lambda _:replace(original,p1=dict(value='released')),original)
    assert value.errors and not value.rows and not value.active
    if not mutation:
        assert error.value is marker
    value.close()
    assert not (tmp_path/A.COMPARISON).exists()


def test_original_error_never_enters_comparison(tmp_path: Path) -> None:
    compared, marker = Comparison(), RuntimeError('original_failure')
    value = consumer(tmp_path,compared)
    def fail(result: Any) -> Any:
        raise marker
    with pytest.raises(RuntimeError) as error:
        value.finish(fail,sample())
    assert error.value is marker and not compared.calls and not value.rows


def test_projection_hidden_field_difference_is_recorded(tmp_path: Path) -> None:
    value = consumer(tmp_path,Comparison())
    value.receiver.rec.side_value = lambda side: dict(value=side['value'])
    old = sample()
    old.p1['score'] = 100
    new = replace(old,p1=dict(old.p1,score=200))
    assert value.finish(lambda _:new,old) is new
    assert value.rows[0]['changed_sides']==['1P']
    assert value.rows[0]['projection_changed_sides']==[]


@pytest.mark.parametrize('target',['side','foreign','registration'])
def test_unserialized_mutation_is_rejected(tmp_path: Path, target: str) -> None:
    value = consumer(tmp_path,Comparison())
    value.receiver.rec.side_value = lambda side: dict(value=side['value'])
    original = sample()
    original.p1['extra'] = []
    def mutate(before: Any, after: Any, lock: Any, tsumo: Any) -> None:
        if target=='side':
            after.p1['extra'].append(1)
        elif target=='foreign':
            after.other['mutable'].append(1)
        else:
            value.receiver.registration['changed'] = True
    value.comparison.consume = mutate
    with pytest.raises(ValueError):
        value.finish(lambda old:old,original)
    assert value.errors and not value.rows


def test_tsumo_cannot_change_during_publication(tmp_path: Path) -> None:
    value = consumer(tmp_path,Comparison())
    def changed(old: Any) -> Any:
        value.receiver.active['pipe'].tsumo_count = lambda side: 11
        return old
    with pytest.raises(ValueError,match='tsumo'):
        value.finish(changed,sample())
    assert not value.comparison.calls and not value.rows


@pytest.mark.parametrize('item',[object(),lambda:None])
def test_snapshot_unknown_type_is_not_silently_dropped(item: Any) -> None:
    with pytest.raises(TypeError,match='unsupported'):
        A.snapshot(item)
