"""人工状態/scoreを原collectへ通し、開始後の正会計原票を取得する診断。"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from types import FunctionType
from typing import Any
import test_anchor as T

ROOT = Path(__file__).resolve().parent
CHAIN_FIRST, SETTLE_FIRST = 96, 100
TOTAL_SCORE = 700


def pipeline(c: Any, mode: str) -> Any:
    pipe = T.pipeline(c, 'good')
    original = pipe.update
    def update(frame_idx: int, time_sec: float, frame: Any) -> Any:
        result = original(frame_idx, time_sec, frame)
        if CHAIN_FIRST <= frame_idx < SETTLE_FIRST:
            result.p1.state = c.BoardState.CHAIN
            result.p1.score = None
        elif frame_idx >= SETTLE_FIRST:
            result.p1.score = TOTAL_SCORE
        return result
    pipe.update = update
    return pipe


def main() -> None:
    real = T.real.__wrapped__()
    settle = sys.modules[real[0].OjamaAccountingTracker.__module__].K_SETTLE_FRAMES
    end = SETTLE_FIRST + T.T.C.STRIDE * (settle + 2)
    paths = [Path(__file__), Path(T.A.__file__), Path(T.__file__), T.A.BASE]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    run = FunctionType(T.run.__code__, dict(vars(T), pipeline=pipeline), argdefs=T.run.__defaults__)
    saved = run(real, ROOT / 'positive_full_v2', 'positive', end)
    pending = saved['accounting']['final_pending_uncapped']
    assert pending['p2'] > 0 and saved['start_anchor']['qualification_frame'] == T.QUALIFIED
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(source_unchanged=True, sources=before, pending=pending,
                  anchor_frame=saved['start_anchor']['qualification_frame'],
                  final_frame=end, actual_video=False, model_evaluated=False, quality_gate_clear=False)
    with (ROOT / 'POSITIVE_PROBE_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result['pending']))


if __name__ == '__main__':
    main()
