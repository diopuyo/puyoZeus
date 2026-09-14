"""実保存で確認した二中継を認証し、旧入口と失敗票を保持する。"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import run_v5 as V5

OLD,V4,WRITER=V5.OLD,V5.V4,V5.WRITER
TRANSPORT_LIMIT=4


def transport(context: Any) -> None:
    journal=context['factory'].provider.journal
    emit=journal.emit
    chain=[]
    terminal=False
    for _ in range(TRANSPORT_LIMIT):
        code=getattr(emit,'__code__',None)
        source=None if code is None else Path(code.co_filename).resolve()
        chain.append(dict(type=type(emit).__name__,qualname=None if code is None else code.co_qualname,
            file=None if source is None else str(source),
            code_sha256=None if code is None else hashlib.sha256(code.co_code).hexdigest(),
            source_sha256=None if source is None or not source.is_file() else hashlib.sha256(source.read_bytes()).hexdigest()))
        if not inspect.isfunction(emit):
            terminal=inspect.ismethod(emit) and emit.__self__ is journal
            break
        emit=inspect.getclosurevars(emit).nonlocals.get('emit')
        if emit is None: break
    packet=dict(emit_chain=chain,terminal_same_journal=terminal,stream_type=type(journal.stream).__name__,
        stream_name=str(getattr(journal.stream,'name',None)),count=journal.count,quality_gate_clear=False)
    with (context['state']['output']/'J_WRITER_TRANSPORT.json').open('x') as stream: json.dump(packet,stream,indent=2)


def create(stack: Any,context: Any) -> Any:
    transport(context)
    state=context['state']
    load=sys.modules['inflight_loader'].load
    client=load('_g2_model_process_client',V4.BRIDGE/'client.py')
    contract=load('_g2_model_process_contract',V4.BRIDGE/'pure_contract.py')
    loader=load('_g2_publication_runtime_loader',OLD.PUB/'runtime_loader.py')
    def supplied(alias: str,path: Any,injection: Any=None) -> Any:
        if Path(path).resolve()==OLD.PUB/'trained_sidecar.py': return client
        if Path(path).resolve()==OLD.PUB/'journal_witness.py':
            original=load('_g2_J_writer_contract_original',WRITER/'writer_contract.py',injection)
            proof=load('_g2_J_writer_contract_v2',WRITER/'writer_contract_v2.py',injection|{'writer_contract':original})
            return load('_g2_J_stream_witness',WRITER/'stream_witness.py',injection|{'writer_contract':proof})
        return load(alias,path,injection)
    modules=loader.modules(state,supplied)
    assert modules['trained_sidecar'] is client,'model_client_injection'
    assert modules['journal_witness'] is sys.modules['_g2_J_stream_witness'],'writer_witness_injection'
    base=type(state['probabilistic_basis_connection']).__init__.__globals__
    config=client.Configuration(base['S'],base['B'].grid,contract.RowInputs)
    return modules['live_session'].Session(stack,context,sys.modules['_g2_hidden_initialization_gate'],
        N(Mode=type(state['probabilistic_tracking_mode'])),contract,config,OLD.FRAMES)


if __name__=='__main__':
    V5.create=create
    raise SystemExit(V5.main())
