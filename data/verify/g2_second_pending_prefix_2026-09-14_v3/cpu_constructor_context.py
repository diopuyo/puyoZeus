"""初期化だけの人工空履歴。画像・物理復帰・実採録の根拠には使わない。"""
from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
FRAME = 35160


class Pipe:
    def update(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError('初期化CPUでは画像updateを呼ばない')


def context(stack: Any, load: Any, parts: Any, output: Path) -> dict:
    original = load('_cpu_actual_J_constructor',ROOT.parent/'g2_atomic_journal_capture_2026-09-09_v1/observer.py')
    journal = original.Recorder.__new__(original.Recorder)
    journal.stream = stack.enter_context((output/'atomic_journal.jsonl').open('x'))
    journal.count,journal.steps,journal.errors,journal.active,journal.closed = 0,0,[],None,False
    journal.history = N(frame=FRAME)
    connection = parts.binding.Connection.__new__(parts.binding.Connection)
    connection.binding = None
    first = parts.mode.Mode.__new__(parts.mode.Mode)
    first.connection,first.native,first.activation,first.arrival_ledger = connection,None,None,None
    first.rows,first.error = 0,None
    first.stream = stack.enter_context(io.StringIO())
    pipe = Pipe()
    journal.pipe,journal.codes = pipe,set()
    state = dict(output=output,probabilistic_basis_connection=connection,probabilistic_tracking_mode=first)
    return dict(state=state,factory=N(provider=N(journal=journal)),pipe=pipe)


def history(stack: Any, load: Any, values: dict) -> Any:
    module = load('_cpu_actual_empty_history',ROOT.parent/'g2_async_projected_evaluation_2026-09-13_v1/early_origin_history.py')
    value = module.EarlyHistory.__new__(module.EarlyHistory)
    value.journal,value.pipe = values['factory'].provider.journal,values['pipe']
    value.frames,value.output = (FRAME,),values['state']['output']
    value.reader,value.stream,value.witness,value.probability = None,None,None,None
    value.scope,value.records,value.probability_records = ExitStack(),[],()
    value.digest,value.probability_digest = hashlib.sha256(),hashlib.sha256()
    value.count,value.last_frame,value.last_steps = 0,FRAME,0
    value.closed,value.sealed,value.transferred,value.transfer_started,value.error = False,True,False,False,None
    with (value.output/module.HISTORY_NAME).open('x'):
        pass
    stack.push(value.close)
    return value
