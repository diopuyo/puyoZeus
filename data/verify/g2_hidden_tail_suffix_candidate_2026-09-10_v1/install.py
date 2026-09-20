"""解決済みの元tail moduleだけに三箇所を設置し、ExitStackで必ず復元する。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import inspect
import consume as C
import source as S
import votes as V

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_hidden_tail_candidate_2026-09-10_v1'


def install(stack: Any, history: Any, patch: Any) -> None:
    """historyは実tail_connection.Hを呼出元が渡す。偽の名前解決はしない。"""
    assert Path(history.__file__).resolve() == ORIGINAL / 'tail_history.py'
    assert history.prepare.__globals__ is vars(history)
    witness = history.T
    assert Path(witness.__file__).resolve() == ORIGINAL / 'tail_witness.py'
    old_source, old_vote, old_consume = witness.source, witness.vote, history.consumed
    assert old_source.__globals__ is vars(witness) and old_vote.__globals__ is vars(witness)
    assert Path(inspect.getsourcefile(old_source)).resolve() == ORIGINAL / 'tail_witness.py'
    derived = C.derive(old_consume)
    def source(control: Any, binding: Any, item: Any, view: Any) -> Any:
        return S.source(old_source, control, binding, item, view)
    def vote(control: Any, binding: Any, item: Any, final: Any,
             raw: Any, proof: Any, view: Any) -> Any:
        return V.vote(old_vote, control, binding, item, final, raw, proof, view)
    patch(stack, witness, 'source', source)
    patch(stack, witness, 'vote', vote)
    patch(stack, history, 'consumed', derived)
