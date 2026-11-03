"""予測候補と到着済みの段だけから確かさを測る。教師は受け取らない。"""
from __future__ import annotations
import numpy as np

CELL_COUNT = 78
SCORE_PER_OJAMA = 70.
UNKNOWN_SEND_WIDTH = 1e9


def candidate_summary(values: list[dict], verified: int, conflict: bool = False) -> dict:
    if not values:
        return dict(origins=0, cells=CELL_COUNT, send_width=UNKNOWN_SEND_WIDTH,
                    verified=verified, stages=0, progress=0., conflict=True)
    boards = np.asarray([v['final'] for v in values])
    sends = [v.get('sent', sum(v['scores'])/SCORE_PER_OJAMA) for v in values]
    stages = max(len(v['scores']) for v in values)
    return dict(origins=len({tuple(v.get('origin', v['final'])) for v in values}),
        cells=int(np.sum(np.any(boards != boards[0], axis=0))),
        send_width=float(max(sends)-min(sends)), verified=verified, stages=stages,
        progress=min(1., verified/max(1, stages)) if not conflict else 0., conflict=conflict)


def confidence_group(summary: dict, threshold: dict) -> int:
    if summary['conflict'] or not summary['origins']:
        return 0
    if summary['origins'] == 1 and summary['cells'] == 0 and summary['send_width'] == 0:
        return 2
    return int(summary['cells'] <= threshold['cells'] and summary['send_width'] <= threshold['send'])


def coefficient(summaries: list[dict], model: dict) -> tuple[float, int, float]:
    if not summaries:
        # 公開済み予測があり未解決連鎖がない場合は、完走後の実確定盤を使っている。
        return 1., 2, 1.
    group = min(confidence_group(s, model['threshold']) for s in summaries)
    progress = min(s['progress'] for s in summaries)
    initial = model['coefficients'][group]
    value = initial+(1.-initial)*progress**model['progress_power']
    return float(value), group, float(progress)
