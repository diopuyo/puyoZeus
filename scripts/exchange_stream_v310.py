"""予測に渡す配送列の採取。終了正解を読まず、次発火/試合末で区切る。"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from scripts.exchange_v310_manifest import OUT
from scripts.exchange_v35_manifest import read, V34, V33
from scripts.exchange_review_v32_data import BASE, video_path
from scripts.exchange_review_io import save
from scripts.exchange_v35_capture import image_sources
from scripts.score_exchange_v35 import next_fire

IMAGE_STEP_SEC = .1


def capture(inp: dict, ocr: dict, obs: list[dict], images: object) -> dict:
    stamp, side = inp['input_sec'], inp['side']
    steps = [s for s in ocr['steps'] if s['side'] == side]
    end = next_fire(inp, obs, ocr['end'], [s['t_sec'] for s in steps])
    records = [r for r in ocr['records'] if r['side'] == side]
    times = [r['t'] for r in records]
    frames, last_image = [], float('-inf')
    initial = next((r for r in reversed(obs) if r['t'] <= stamp), {})
    for row in obs:
        now = row['t']
        if not stamp < now < end:
            continue
        index = bisect_right(times, now)-1
        score = records[index] if index >= 0 else None
        if score is None or score['t'] <= stamp:
            continue
        sources = []
        if score['score'] is not None and now-last_image >= IMAGE_STEP_SEC:
            sources = images.boards(now, now, side, offsets=(-1/images.fps, 0.))
            for i, source in enumerate(sources):
                source['classifier'] = 'vote' if i % 2 else 'median'
            last_image = now
        frames.append(dict(row, normal_score=score['score'] is not None, score_t=score['t'], sources=sources))
    return dict(initial_next=initial.get('next'), frames=frames,
        formulas=[s for s in steps if stamp < s['t_sec'] < end],
        scores=[r for r in records if stamp < r['t'] < end],
        score_prefix=[r for r in records if r['t'] <= stamp],
        formula_prefix=[s for s in steps if s['t_sec'] <= stamp], end=end,
        note='連鎖最終段は識別しない。全通常得点/盤面を配送し、採点側だけが最終段を指定。')


def collect_new() -> None:
    inputs = read(OUT/'inputs.json')
    for video in sorted({i['video'] for i in inputs}):
        images = image_sources(video_path(video))
        for inp in sorted([i for i in inputs if i['video'] == video], key=lambda i: i['input_sec']):
            target = OUT/f'streams/{video}_{inp["event_id"]}.json'
            if target.exists():
                continue
            folder = OUT/f'video_{video}/game_{inp["game"]}'
            ocr, obs = read(folder/f'holdout_{video}_ocr.json'), read(folder/'observations.json')
            save(target, capture(inp, ocr, obs[inp['side']], images))
            print('配送採取', video, inp['event_id'], flush=True)
        images.cap.release()


def collect_video80() -> None:
    inputs = read(V34/'prediction_inputs.json')
    probes = {r['event_id']: r for r in read(V33/'truth_probe.json')}
    ocr, obs = read(V33/'ocr.json'), read(V34/'observations_80.json')
    images = image_sources(video_path(80))
    for inp in inputs:
        probe = probes[inp['event_id']]
        inp.update(video=80, game=7, side=probe['side'], trigger_sec=probe['trigger'])
        target = OUT/f'streams/80_{inp["event_id"]}.json'
        if not target.exists():
            save(target, capture(inp, ocr, obs[inp['side']], images))
    save(OUT/'video80_inputs.json', inputs)
    images.cap.release()


if __name__ == '__main__':
    collect_new()
    collect_video80()
