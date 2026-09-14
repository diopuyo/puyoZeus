"""実v12の保存済みraw/CNN/confirmed/PBを比較する。欠測を仮値で埋めない。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
LIVE=ROOT.parent/'video38_history_publication_probe_live_2026-09-11_v12'


def records() -> tuple[dict,list[dict]]:
    source=json.loads((LIVE/'LIVE_EMPTY_RESET.json').read_bytes())
    waits=[row for row in source['recovery'] if row['kind']=='reset_baseline_wait']
    reset=next(row['frame'] for row in source['events'] if row['stage']=='reset_returned')
    assert reset==35160 and [r['frame'] for r in waits]==list(range(reset,35175,2))
    return dict(reset_frame=reset,deadline=reset+14),waits


def differences(left: Any,right: Any) -> list[list[int]]|None:
    if left is None or right is None: return None
    return [[r,c,left[r][c],right[r][c]] for r in range(13) for c in range(6) if left[r][c]!=right[r][c]]


def report() -> dict:
    clock,waits=records()
    rows=[]
    for wait in waits:
        channels=wait['channels']
        grid=lambda name:None if channels[name] is None else channels[name]['grid']
        raw,confirmed,cnn=wait['raw'],grid('confirmed'),grid('cnn')
        pb=channels['probability']
        cells=None if pb is None else pb.get('cells')
        rows.append(dict(frame=wait['frame'],journal_token=wait['journal_token'],scope=wait['scope'],
            state=channels['state'],saved_reason=wait['reason'],raw_cnn_diff=differences(raw,cnn),
            raw_confirmed_diff=differences(raw,confirmed),actual_probability_present=cells is not None,
            hidden_probability=None if cells is None else cells[0],
            input_values_fabricated=False,full_gate_flags_available=False))
    return dict(**clock,rows=rows,quality_gate_clear=False,actual_factory_called=False)


if __name__=='__main__':
    result=report()
    with (ROOT/'ACTUAL_CHANNELS.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(reset_frame=result['reset_frame'],deadline=result['deadline'],
        rows=[{k:v for k,v in r.items() if k in ('frame','state','saved_reason','raw_confirmed_diff','actual_probability_present')}
              for r in result['rows']]),ensure_ascii=False))
