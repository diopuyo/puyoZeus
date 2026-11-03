"""配送順に終了盤を読み、最初の可読観測で残存数だけを補正する。"""
from __future__ import annotations

from bisect import bisect_right
import json
from time import perf_counter
from scripts.audit_exchange_counts_v33 import OUT as V33
from scripts.audit_exchange_cells_v34 import OUT
from scripts.exchange_review_v32_data import BASE, video_path
from scripts.exchange_counts_v33_collect import ImageSources
from scripts.exchange_review_io import save
from src.exchange_count_correction import RemainingCorrection
from src.exchange_count_prediction import LOW_SPEC_RATIO

MS = 1000


def correct(event: dict, observations: list[dict], records: list[dict], images: ImageSources,
            native: object) -> dict:
    wrapper = RemainingCorrection(event['input_sec'])
    stamps, attempted = [r['t'] for r in records], []
    for raw in observations:
        stamp = raw['t']
        if stamp <= event['input_sec']:
            continue
        index = bisect_right(stamps, stamp)-1
        ocr = records[index] if index >= 0 else None
        if ocr is None or ocr['score'] is None or ocr['t'] <= event['input_sec']:
            continue
        sources = images.boards(stamp, stamp, event['side'], offsets=(-1/images.fps, 0.))
        for i, source in enumerate(sources):
            source['classifier'] = 'vote' if i % 2 else 'median'
        began = perf_counter()
        result = wrapper.observe(stamp, raw['state'], True, sources, event['colors'], native,
                                 allow_visual_stable=True)
        elapsed = (perf_counter()-began)*MS+sum(s['image_compute_ms'] for s in sources)
        attempted.append(stamp)
        if result:
            result.update(event_id=event['event_id'], low_spec_ms=elapsed*LOW_SPEC_RATIO,
                publish_sec=stamp+elapsed*LOW_SPEC_RATIO/MS, attempted=attempted,
                evidence_times=result['evidence_times']+[ocr['t']], sources=sources)
            return result
    return dict(event_id=event['event_id'], remaining=None, attempted=attempted, reason='次の発火まで可読盤なし')


def main() -> None:
    import puyo_core
    observations = json.loads((OUT/'observations_80.json').read_text())
    records = json.loads((V33/'ocr.json').read_text())['records']
    items = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']
    predictions = json.loads((OUT/'predictions.json').read_text())['rows']
    real = {r['event_id'] for r in predictions if r['variants']}
    items = [i for i in items if i['event_id'] in real]
    images, output = ImageSources(video_path(80)), []
    for item in items:
        stamp, side = item['fire']['t_sec'], item['side']
        limit = next((i['fire']['t_sec'] for i in items if i['side'] == side and i['fire']['t_sec'] > stamp), float('inf'))
        event = dict(event_id=item['event_id'], input_sec=stamp, side=side,
                     colors=item['fire']['exchange_input']['colors'])
        result = correct(event, [r for r in observations[side] if stamp < r['t'] < limit],
                         [r for r in records if r['side'] == side], images, puyo_core)
        output.append(result)
        print('残存補正', item['event_id'], result.get('input_sec'), result['remaining'], flush=True)
    images.cap.release()
    save(OUT/'corrections.json', output)


if __name__ == '__main__':
    main()
