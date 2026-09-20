"""原列挙と保存実rawを再用する局所支持試験。原update/会計公開ではない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
PROJECT, VERIFY = ROOT.parents[2], ROOT.parent
sys.path.insert(0,str(PROJECT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'))
from src import puyo_core_bridge as CORE
import path_support as P


def inputs() -> tuple[Any,Any]:
    live = VERIFY/'video38_history_publication_probe_live_2026-09-10_v7'
    rows = [json.loads(line) for line in (live/'directional_history.jsonl').read_text().splitlines()]
    first = next(r for r in rows if r['scope']['frame_idx']==35002)
    later = next(r for r in rows if r['scope']['frame_idx']==35054)
    return first['grid_after'],later['raw_capture']['raw']['grid']


def test_actual_two_hand_unique_without_unknown_overwrite() -> None:
    before,raw = inputs()
    saved = deepcopy((before,raw))
    hits = P.infer(CORE,before,(5,4),(2,2),raw)
    assert len(hits)==1
    h = hits[0]
    assert h.raw[0][0]==10 and h.prefix[0][0]==h.final[0][0]==4
    assert sum(c in P.COLORS for row in h.prefix for c in row)==63
    assert sum(c in P.COLORS for row in h.final for c in row)==65
    assert not h.probability_assigned and not h.current_permission and not h.accounting_permission
    assert (before,raw)==saved


def test_wrong_pair_is_not_supported() -> None:
    before,raw = inputs()
    assert not P.infer(CORE,before,(5,4),(3,3),raw)


@pytest.mark.parametrize('case', ('visible_unknown','boolean','wrong_shape'))
def test_invalid_observation_rejected(case: str) -> None:
    before,raw = inputs()
    if case == 'visible_unknown': raw[1][0]=10
    if case == 'boolean': raw[1][0]=True
    if case == 'wrong_shape': raw.pop()
    with pytest.raises(ValueError): P.infer(CORE,before,(5,4),(2,2),raw)


def test_known_hidden_contradiction_rejected() -> None:
    before,raw = inputs()
    raw[0][0]=3
    assert not P.infer(CORE,before,(5,4),(2,2),raw)
