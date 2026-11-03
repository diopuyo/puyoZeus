"""最終勝敗を使わず、発火から固定5秒後の公開B3の変化を回帰する。"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any
import numpy as np
from scripts.exchange_review_io import pickles, save
from scripts.exchange_review_v3_input import Inputs, schedule
from scripts.exchange_review_seed import PrefixSeeds
from scripts.exchange_review_v3_compute import compute_row

TARGET_SEC = 5.
VIDEOS = (29, 32, 34, 39, 51, 61, 65, 79, 92, 94)


class B3:
    def __init__(self, path: Path, fps: float) -> None:
        self.rows = [dict(t=r['frame'] / fps, game=r.get('public_game', r['game']), p=r['probability'])
                     for r in pickles(path)]
        self.times = [r['t'] for r in self.rows]

    def at(self, t: float, game: int) -> float | None:
        index = bisect_right(self.times, t) - 1
        row = self.rows[index] if index >= 0 else None
        return row['p'] if row and row['game'] == game else None


def stats(predicted: list, actual: list) -> dict:
    x, y = np.array(predicted, dtype=float), np.array(actual, dtype=float)
    if not len(x):
        return dict(n=0, correlation=None, mae=None, rmse=None, sign_accuracy=None)
    correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
    return dict(n=len(x), correlation=correlation, mae=float(np.mean(np.abs(x - y))),
        rmse=float(np.sqrt(np.mean((x - y) ** 2))),
        sign_accuracy=float(np.mean(np.sign(x) == np.sign(y))))


def sample(event: dict, inputs: Inputs, b3: B3, video: int) -> dict | None:
    choice = event.get('recommendations', {}).get('fast')
    game, stamp = event['game_idx'], event['input_sec']
    target = event['fire_sec'] + TARGET_SEC
    before, after = b3.at(stamp, game), b3.at(target, game)
    if not choice or before is None or after is None or event['fast_ready_sec'] >= target:
        return None
    pending = inputs.pending(stamp, game)['pending']
    now = pending[1] - pending[0]
    post = (1 if event['receiver'] == '1P' else -1) * choice['balance']
    return dict(video=video, game=game, event=event['event_id'], fire_sec=event['fire_sec'],
        input_sec=stamp, ready_sec=event['fast_ready_sec'], target_sec=target,
        deferred=event['deferred'], x=post - now, y=after - before, b3_now=before, b3_after=after)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(output: Path, table: Any, native: Any) -> dict:
    samples = []
    for video in VIDEOS:
        cache = output / f'fit_{video}.json'
        if cache.exists():
            samples.extend(json.loads(cache.read_text()))
            continue
        path = output / 'fit_input' / f'video_{video}'
        inputs = Inputs(path)
        fps = json.loads((path / 'video_receipt.json').read_text())['fps']
        b3 = B3(path / 'publications.pickle', fps)
        prefix, part = PrefixSeeds(path / 'inputs.jsonl.gz', table), []
        scheduled = schedule(inputs)
        for i, row in enumerate(scheduled):
            if row['game_idx'] <= 0:
                continue
            event = compute_row(row, prefix, inputs, native, recursive=False)
            item = sample(event, inputs, b3, video)
            if item:
                part.append(item)
            if i % 100 == 0:
                print(f'当てはめ 動画{video}: {i}/{len(scheduled)}', flush=True)
        save(cache, part)
        samples.extend(part)
    x, y = np.array([r['x'] for r in samples]), np.array([r['y'] for r in samples])
    slope = float(x @ y / (x @ x)) if x @ x else 0.
    # 切片0なので左右反転で予測差も反転する。非負制約や任意倍率は追加しない。
    model = dict(slope=slope, intercept=0., videos=list(VIDEOS), excluded_video=80,
        samples=len(samples), games=len({(r['video'], r['game']) for r in samples}),
        target_sec=TARGET_SEC, deferred_samples=sum(r['deferred'] for r in samples),
        fit=stats((slope * x).tolist(), y.tolist()), baseline=stats([0.] * len(y), y.tolist()),
        definition='ΔB3 = slope × (予測収支1P視点 − 現在の予告収支)。目標は発火+5秒。遅延再計算は計算時B3を基準。')
    save(output / 'model.json', model)
    write_csv(output / 'fit_samples.csv', samples)
    return model
