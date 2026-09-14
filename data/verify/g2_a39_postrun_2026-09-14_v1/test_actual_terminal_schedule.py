"""原A39終了原票を元資格/Scheduled.completedへ供給。basis初期化は対象外。"""
from collections import deque
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

VERIFY = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(VERIFY/'g2_m1_completion_candidate_2026-09-12_v1'),
               str(VERIFY/'g2_legacy_m1_compatibility_2026-09-14_v1')]
from test_scheduled_session import module
import compatibility as C

OUTPUT = VERIFY/'video38_second_prefix_candidate_v39'
FRAME, END, TAIL_ROWS = 36886, 36900, 16


def read(name: str) -> Any:
    return json.loads((OUTPUT/name).read_bytes())


def tail(name: str) -> list:
    with (OUTPUT/name).open() as stream:
        return [json.loads(line) for line in deque(stream, maxlen=TAIL_ROWS)]


def test_actual_tail_has_no_false_M1_request(module: Any, monkeypatch: Any) -> None:
    saved = read('BELIEF_M1_SESSION.json')
    context = tail('provisional_context.jsonl')[-1]
    assert context['frame_idx'] == FRAME
    steps = [r for r in tail('atomic_journal.jsonl') if r['kind']=='step' and r['frame_idx']==FRAME]
    ledgers = [read(name)['receipt']['last_live_ledger']
        for name in ('FIRST_TERMINAL_STATUS.json','SECOND_INACTIVE_END_STATUS.json')]
    modes = [N(error=None, closed=False, connection=N(binding=object()), native=N(pending=()),
               arrival_ledger=N(**ledger)) for ledger in ledgers]
    journal = object()
    session = N(error=None, basis=lambda:None, mode=modes[1], saved=[], schedule_rows=0,
        schedule=module.S.Schedule(last=saved['schedule']['last'], accepted=tuple(saved['schedule']['accepted'])),
        schedule_stream=io.StringIO(),
        state=dict(provisional_context_observer=N(active=None, errors=[], rows=[context]),
                   probabilistic_tracking_mode=modes[0]),
        witness=N(journal=journal, pair=lambda frame:steps),
        evaluation_flags=N(journal=journal,closed=False,error=None,latest=saved['evaluation_flags_latest']))
    monkeypatch.setattr(module.S, 'END', END)
    C.completed(module.Session.completed, session, FRAME)
    packet = json.loads(session.schedule_stream.getvalue())
    assert packet['action'] == 'WAIT' and not packet['saved']
    assert session.schedule.last == FRAME and session.schedule.pending is None and session.saved == []
    assert '1P:match_active' in packet['reasons'] and '2P:match_active' in packet['reasons']
