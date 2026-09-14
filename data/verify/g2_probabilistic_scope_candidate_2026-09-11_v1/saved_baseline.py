"""実v12の同run J・PB・復帰票を結合し、私有確率基準を再構成する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import belief as B

ROOT=Path(__file__).resolve().parent
LIVE=ROOT.parent/'video38_history_publication_probe_live_2026-09-11_v12'
FRAME,SIDE=35164,'1P'
OUTPUT=ROOT/'SAVED_BASELINE.json'


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def unique(path: Path,kind: str|None=None) -> Any:
    found=[]
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['frame_idx']>FRAME: break
            if row['frame_idx']==FRAME and row['side']==SIDE and (kind is None or row['kind']==kind):
                found.append(row)
    B.require(len(found)==1,'saved_row_not_unique:'+path.name)
    return found[0]


def join() -> tuple[Any,Any,Any]:
    reset=read(LIVE/'LIVE_EMPTY_RESET.json')
    waits=[r for r in reset['recovery'] if r['kind']=='reset_baseline_wait' and r['frame']==FRAME]
    B.require(len(waits)==1 and waits[0]['reason']=='unknown','saved_wait')
    wait=waits[0]
    pb=unique(LIVE/'hidden_probability.jsonl')
    journal=unique(LIVE/'atomic_journal.jsonl','step')
    B.require(journal['token']==wait['journal_token'] and journal['exception'] is None,'saved_J_identity')
    source,run,epoch,pipe,sm,generation,side=wait['scope']
    B.require((source,run,epoch,pipe,generation,side)==(journal['source_id'],journal['run_id'],
        journal['software_reset'],journal['pipe_object_id'],journal['generation']['reset_epoch'],journal['side']),'saved_scope')
    B.require(journal['generation']==journal['generation_after']
        and journal['generation']['side']==side,'saved_generation_changed')
    B.require((pb['source_id'],pb['run_id'],pb['time_sec'],pb['side'])
        ==(source,run,FRAME/60,side),'saved_PB_identity')
    B.require(pb['state_value']=='stable' and pb['instrumentation_errors']==[],'saved_PB_state')
    B.require(pb['probability']['cells']==wait['channels']['probability']['cells'],'saved_PB_same_call')
    B.require(pb['confirmed']['grid']==wait['channels']['confirmed']['grid']==wait['raw'],'saved_grids')
    B.require(reset['events'][0]['old_integer_present'] is True,'saved_old_integer')
    return reset,wait,pb


def main() -> None:
    start=time.perf_counter()
    names=('LIVE_EMPTY_RESET.json','hidden_probability.jsonl','atomic_journal.jsonl')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pins={name:sha(LIVE/name) for name in names}
    reset,wait,capture=join()
    confirmed=B.Board.from_dict({'grid':wait['raw']})
    probability=B.ProbabilisticBoard.from_board(confirmed)
    for r in range(B.BOARD_ROWS):
        for c in range(B.BOARD_COLS):
            probability.set_distribution(r,c,dict(capture['probability']['cells'][r][c]))
    state=B.establish(tuple(wait['scope']),FRAME,reset['recovery'][-1]['frame'],confirmed,probability)
    B.require(len(state.worlds)==343 and not state.source_producer_connected,'saved_support')
    B.require(pins=={name:sha(LIVE/name) for name in names},'saved_input_changed')
    old=reset['events'][0]['old_owner']['current']
    result=dict(frame=FRAME,support=len(state.worlds),weight_sum=sum(w.weight for w in state.worlds),
        old_integer_colors=sum(1 for row in old['grid'] for c in row if c in B.PIECE_COLORS),
        old_integer_unchanged=True,source_sha256=pins,same_run_saved_J_PB_join=True,
        actual_live_factory_called=False,integer_baseline_established=False,
        quality_gate_clear=False,seconds=time.perf_counter()-start)
    with OUTPUT.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False))


if __name__=='__main__': main()
