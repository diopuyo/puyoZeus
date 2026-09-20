"""実採録・実RowStreamを使い、人工2更新で次入口と保存の接続を検査。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import numpy as np
import pytest
import collector_continuous as C
import metadata_tail as M

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
FIRST, STRIDE, FPS = 34772, 2, 60


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def real() -> Any:
    base = load('_continuous_original_loader',REPO/'scripts/diagnose_video38_confirmed_collapse_v1.py')
    collector = base.load_collector()
    bounded = load('_continuous_bounded',ROOT.parent/'g2_collector_metadata_bounded_2026-09-09_v1/bounded.py')
    return collector,bounded


def pipeline(c: Any) -> Any:
    grid = np.zeros((13,6),dtype=np.int8)
    grid[12,0] = 1
    board = c.Board.from_list(grid.tolist())
    side = lambda: N(confirmed_board=board,state=c.BoardState.STABLE,score=0,
                     next_pair=(1,2),dnext_pair=(3,4),chain_event=None)
    class Pipeline:
        def __init__(self) -> None:
            self.calls: list[Any] = []
        def update(self, frame_idx: int, time_sec: float, frame: Any) -> Any:
            self.calls.append(('update',frame_idx))
            return N(p1=side(),p2=side(),is_match_active=True)
        def tsumo_count(self, side: str) -> int:
            self.calls.append(('tsumo',side))
            return 0
    return Pipeline()


def run(c: Any, b: Any, output: Path, enabled: bool) -> Any:
    output.mkdir()
    state = dict(output=output)
    pipe = pipeline(c)
    cap = N(read=lambda: (True,np.zeros((1080,1920,3),dtype=np.uint8)))
    with ExitStack() as stack:
        b.install(stack,c,None,state,enabled=enabled)
        if enabled: M.install(stack,state)
        loop = C.build(c,b.O.SOURCE)
        stack.callback(loop.generator.close)
        loop.RecognitionPipeline = type(pipe)
        for frame in (FIRST,FIRST+STRIDE):
            if enabled and frame>FIRST: M.completed(state,frame-STRIDE)
            loop.collect_lean(cap,pipe,frame,1,STRIDE,FPS)
        if enabled: M.completed(state,FIRST+STRIDE)
        acc = loop.runtime_state['acc']
        saved = {f.name:b.O.serial(getattr(acc,f.name)) for f in fields(acc) if f.name!='wons'}
    return state,saved,pipe.calls


def test_original_downstream_saved_and_noninterference(real: Any, tmp_path: Path, monkeypatch: Any) -> None:
    c,b = real
    monkeypatch.setattr(b.O,'FIRST',FIRST)
    monkeypatch.setattr(b.O,'LAST',FIRST+STRIDE)
    off,old,calls = run(c,b,tmp_path/'off',False)
    state,new,newcalls = run(c,b,tmp_path/'on',True)
    assert old==new and calls==newcalls and len(calls)==6
    b.finish(state)
    b.verify(tmp_path/'on')
    rows = [json.loads(line) for line in (tmp_path/'on'/b.O.ENTRIES).read_text().splitlines()]
    assert len(rows)==4 and sum(len(r['appends']) for r in rows)==2
    assert state[M.KEY].count==4 and len(state[M.KEY].rows)==2
    assert set(vars(state[b.KEY].rows))=={'stream','count','append_count'}


def test_missing_previous_side_refuses(real: Any, tmp_path: Path, monkeypatch: Any) -> None:
    c,b = real
    monkeypatch.setattr(b.O,'FIRST',FIRST)
    monkeypatch.setattr(b.O,'LAST',FIRST+STRIDE)
    state = dict(output=tmp_path)
    with ExitStack() as stack:
        b.install(stack,c,None,state,enabled=True)
        M.install(stack,state)
        with pytest.raises(ValueError,match='metadata_missing'): M.completed(state,FIRST)


def test_source_edit_refuses(tmp_path: Path) -> None:
    with pytest.raises((AssertionError,FileNotFoundError)): C.extracted(tmp_path/'not_original.py')
