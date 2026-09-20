"""元PBを保存しつつ隠し分布だけを別生成する境界。実接続は別試験。"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent/'g2_hidden_two_hand_candidate_2026-09-10_v1'),
    str(ROOT.parent/'g2_hidden_tail_candidate_2026-09-10_v1'),
    str(ROOT.parents[2]/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30')]
from src.board import Board
from src.probabilistic_board import ProbabilisticBoard
import conditional_current as C


def boards() -> tuple[Any,...]:
    before = Board()
    inferred = [[0]*6 for _ in range(13)]
    for r in range(1,13):
        before.set(r,0,5)
        inferred[r][0]=5
    inferred[0][0]=4
    visible = tuple(tuple(row) for row in before.to_dict()['grid'])
    return ProbabilisticBoard.from_board(before),visible,tuple(map(tuple,inferred))


def test_conditional_pb_does_not_overwrite_original() -> None:
    original,visible,inferred = boards()
    frozen = C.cells(original)
    output,receipt = C.probability(ProbabilisticBoard,original,visible,inferred)
    assert output is not original and receipt==frozen==C.cells(original)
    assert original.cell(0,0).probs=={0:1.0} and output.cell(0,0).probs=={4:1.0}
    assert C.cells(output)[1:]==frozen[1:]


def test_visible_pb_conflict_rejected() -> None:
    original,visible,inferred = boards()
    original.set_certain(12,0,2)
    with pytest.raises(ValueError): C.probability(ProbabilisticBoard,original,visible,inferred)


def current() -> tuple[Any,...]:
    original,visible,inferred = boards()
    output,_ = C.probability(ProbabilisticBoard,original,visible,inferred)
    scope = ('source','run',2,'1P')
    proof = dict(scope=scope,frame=34856,clock=34856/60,action=2,sm=visible,
        inferred_final=inferred,integer_anchor='null',conditional_PB=C.cells(output),
        conditional_not_recognition_confidence=True,original_PB_unchanged=True)
    value = C.ConditionalCurrent(scope,34856,34856/60,2,visible,inferred,C.cells(output)[0],
        'null',C.encoded(proof))
    return value,N(current=None,action=2),N(scope=scope,current=visible,hidden_current=value)


def test_correspondence_is_not_integer_permission() -> None:
    value,state,binding = current()
    assert C.compatible(state,binding) and C.intact(value)
    assert not value.integer_current_permission and not value.accounting_permission
    with pytest.raises(TypeError): C.ConditionalCurrent(**asdict(value))


@pytest.mark.parametrize('field',('frame','clock','action','sm_grid','inferred_grid','hidden','integer_anchor'))
def test_mutated_certificate_rejected(field: str) -> None:
    value,state,binding = current()
    changed = {'frame':34858,'clock':0.0,'action':1,'sm_grid':(),
        'inferred_grid':(),'hidden':(),'integer_anchor':'{}'}
    binding.hidden_current = replace(value,**{field:changed[field]})
    assert not C.compatible(state,binding)
