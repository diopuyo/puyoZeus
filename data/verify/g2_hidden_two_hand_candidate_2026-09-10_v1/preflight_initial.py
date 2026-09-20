"""重い全接続前に原constructor/SMで人工初期条件を検証する。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import prefix_inputs as I

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'g2_hidden_current_exit_probe_2026-09-10_v1'))
import probe_assets as A


def main() -> None:
    started = time.perf_counter()
    board,state,(build,flags),_,_ = A.modules()
    sm = build(**flags)
    receipt = I.initial_fixture(SimpleNamespace(_sm_1p=sm),
        SimpleNamespace(controller=SimpleNamespace(history={})))
    rows: list[Any] = []
    for frame in I.FRAMES:
        if frame>=I.FIRST: break
        signals = state.DetectorSignals(time_sec=frame/I.FPS,is_match_active=True,
            cnn_board=board.Board.from_list(receipt['initial_confirmed']),
            effect_gate_window_active=False)
        result = sm.update(frame,signals)
        assert A.grid(result.confirmed_board)==receipt['initial_confirmed']
        rows.append(dict(frame=frame,state=result.state.value))
    assert rows[-1]['state']=='stable'
    value = dict(initial=receipt,rows=rows,seconds=time.perf_counter()-started,
        original_SM_only=True,full_pipeline=False,quality_gate_clear=False)
    with (ROOT/sys.argv[1]).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
    print(dict(updates=len(rows),seconds=value['seconds'],natural_stable=True))


if __name__=='__main__':
    main()
