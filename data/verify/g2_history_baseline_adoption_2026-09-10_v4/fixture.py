"""原J enqueue/原Fifo/原Sを使う人工transport。認識fullではない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
FIRST,BASELINE,SIDE = 34778,34796,'1P'
NATIVE_FIRST,NATIVE_LAST = 32494,36298


def load(name: str, path: Path) -> Any:
    if name in sys.modules:
        raise RuntimeError(name)
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load('_adoption_baseline_fixture',VERIFY/'g2_history_baseline_entry_independent_2026-09-10_v1/probe.py')
M = P.modules()
J = load('_adoption_original_journal',VERIFY/'g2_atomic_journal_capture_2026-09-09_v1/observer.py')
W = load('_adoption_witness_unit',ROOT/'witness.py')
A = load('_adoption_binding_unit',ROOT/'adoption.py')


@dataclass
class Candidate:
    available_frame: int = FIRST
    first_support_frame: int = FIRST-2
    sequence_number: int = 39


def journal(item: Any) -> Any:
    rec = object.__new__(J.Recorder)
    rec.source_id,rec.run_id = item.view.scope[:2]
    rec.selected = {(f,SIDE) for f in (FIRST,BASELINE)}
    rec.enqueues,rec.errors,rec.update_before = 0,[],{}
    rec.fifo,rec.rows = J.Fifo(),[]
    rec.emit = lambda row:rec.rows.append(deepcopy(row))
    rec.epoch = lambda pipe,side:2
    rec.scope = lambda pipe,side,frame,clock:dict(source_id=rec.source_id,run_id=rec.run_id,
        side=side,pipe_object_id=id(pipe),frame_idx=frame,time_sec=clock,
        generation=dict(side=side,reset_epoch=2,action_revision=49))
    rec.history = SimpleNamespace(emit=lambda row:rec.rows.append(deepcopy(row)))
    return rec


def fixture() -> Any:
    item = P.fixture(M,P.evidence(),False)
    item.pipe._sm_1p = item.sm
    item.pipe._pending_tsumo_1p = item.view.queue
    item.pipe._tsumo_count_1p = item.pipe.counter
    item.pipe._first_move_sec_1p = 545.5
    item.pipe._last_consumed_color_1p = None
    item.pipe._last_seen_next_1p = (2,5)
    item.pipe._landing_pending_1p = None
    rec = journal(item)
    provider = object.__new__(M.V.Provider)
    provider.journal,provider.enqueues = rec,{}
    provider._parts = SimpleNamespace(C=M.C,T=M.T,H=M.C.H,O=SimpleNamespace(MotionCandidate=Candidate))
    provider.link = SimpleNamespace(latest={},attached=True,error=None,
        adapter=SimpleNamespace(enabled=True,native=SimpleNamespace(FIRST_FRAME=NATIVE_FIRST,LAST_FRAME=NATIVE_LAST)))
    provider.link.current = lambda view:provider.link.latest[SIDE]
    provider.link.basis = lambda view:((2,5),(5,5))
    def attach(stack: Any, control: Any) -> None:
        old = rec.emit
        def emitted(row: Any) -> None:
            old(row)
            if row.get('kind')=='enqueue':
                provider.enqueues[SIDE] = deepcopy(row)
        rec.emit = emitted
        stack.callback(setattr,rec,'emit',old)
    provider.attach = attach
    item.control.provider = provider
    item.provider,item.rec = provider,rec
    item.witness = A.bind(W,provider,item.control)
    return item


def enqueue(item: Any, frame: int, append: bool) -> None:
    def native(pipe: Any, side: str, clock_frame: int, clock: float, active: bool, pair: Any) -> None:
        if append:
            pipe._pending_tsumo_1p.append((4,3))
            pipe._last_consumed_color_1p = pipe._pending_tsumo_1p[-1]
        item.provider.link.latest[SIDE] = dict(pipe=pipe,frame=frame,epoch=2,segment='scope-segment',
            queue=pipe._pending_tsumo_1p,refs=tuple(pipe._pending_tsumo_1p),committed=append,
            reason='occurrence_committed' if append else 'await_motion',baseline_frame=FIRST,
            pending=None,armed_at=None,event=Candidate())
    original = J.Recorder.wrap_enqueue(item.rec,native)
    original(item.pipe,SIDE,frame,frame/60,True,(2,5))


def view(item: Any) -> Any:
    owner = item.provider.owner(item.pipe,SIDE,2)
    scope = item.view.scope[:5]+(2,SIDE)
    return replace(item.view,scope=scope,refs=owner['refs'],tokens=tuple(owner['tokens']),
        next_pair=(2,5),dnext_pair=(5,5),quiet=True,added=())
