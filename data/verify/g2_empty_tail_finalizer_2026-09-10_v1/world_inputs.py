"""凍結した76保存だけを読み、world検証用の元構造を渡す。"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent/'g2_empty_tail_next_2026-09-10_v1/prefix_cpu_v2'
NAMES=('directional_history.jsonl','atomic_journal.jsonl','COMBINED_HIDDEN.json',
    'POSTCOMMIT_CONSUMER_ROWS.json','RESULT.json')


def read(name: str) -> Any:
    with (RUN/name).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream] if name.endswith('.jsonl') else json.load(stream)


def inputs() -> Any:
    result=read('RESULT.json')
    assert result['exit_code']==0 and result['guards_unchanged'] and result['updates']==76
    index=json.loads((RUN/'INDEX.json').read_bytes())
    assert all(sha256((RUN/name).read_bytes()).hexdigest()==index[name] for name in NAMES)
    hidden=read('COMBINED_HIDDEN.json')
    histories=read('directional_history.jsonl')
    assert not result['state']['origins'] and not result['state']['debts']
    allowed={'hidden_two_hand_prefix_history/v1','hidden_single_tail_history/v1','hidden_empty_tail_next_history/v1'}
    assert all(r['prepared'] is None or r['prepared']['kind'] in allowed for r in histories)
    # この非発火fixtureにはADMISSION保存がない。登録行の網羅性は実factory結合で別検査する。
    return dict(history_rows=histories,journal_rows=read('atomic_journal.jsonl'),
        conditional_rows=[],hidden_events=hidden['events'],
        hidden_history=hidden['history'],hidden_lifetime=hidden['lifetimes'],
        outer_rows=read('POSTCOMMIT_CONSUMER_ROWS.json'))
