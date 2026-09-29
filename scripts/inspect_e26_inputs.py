"""E26入力に途中観測があるか、物理状態遷移と原映像を先に監査する。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import cv2

from src.exchange_event_record import read_records
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO


def inspect(source: str) -> dict:
    """確定盤面を途中認識の代用品にせず、保存属性と遷移を列挙する。"""
    counts, transitions, rows = Counter(), Counter(), []
    previous = [None, None]
    for row in read_records(Path('logs/e16/records')/f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        result, _, _, stamp, *_ = row['args']
        for idx, side in enumerate((result.p1, result.p2)):
            state = side.state.name
            counts[state] += 1
            counts['cnn_board'] += getattr(side, 'cnn_board', None) is not None
            if previous[idx] != state:
                transitions[f'{previous[idx]}->{state}'] += 1
                if source == 'review' and idx == 1 and 2751 <= stamp <= 2772:
                    rows.append(dict(t_sec=stamp, previous=previous[idx], state=state))
            previous[idx] = state
    return dict(counts=dict(counts), transitions=dict(transitions), scene=rows)


def main() -> None:
    """入力監査を再現可能なJSONとして保存する。"""
    result = {s: inspect(s) for s in (*SOURCES, 'zenchi', 'review')}
    save_json(Path('logs/e26/INPUT_AUDIT.json'), result)
    metadata = {}
    for source in (*SOURCES, 'zenchi'):
        video = VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4' if source in SOURCES else ZENCHI_VIDEO
        capture = cv2.VideoCapture(str(video))
        metadata[source] = dict(width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT), fps=capture.get(cv2.CAP_PROP_FPS))
        capture.release()
    save_json(Path('logs/e26/VIDEO_METADATA.json'), metadata)
    print(metadata, flush=True)


if __name__ == '__main__':
    main()
