"""同じ保存通知を全件評価し、指定公開フレームでB7との誤差を比較する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.replay_exchange_event_20260926 import static_builder
from src.exchange_event_evaluator import FileExchangeModels
from src.exchange_event_m0 import FileM0Predictor
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_record import read_records
from src.phase_j.live_evaluation import SplitExchangeOverlay, sampled_ema

PERCENTILES = (50, 95, 99)


def distribution(values: list[float]) -> dict[str, Any]:
    return dict(count=len(values), maximum=max(values, default=None),
        **{f'P{p}': float(np.percentile(values, p)) if values else None for p in PERCENTILES})


def physical_records(overlay: Any) -> list[dict]:
    """勝率の計算履歴を除き、全通知の状態機械を独立に照合する。"""
    records = [asdict(record) for record in overlay.tracker.records]
    for record in records:
        record.pop('values')
    return records


def make_pair(record: Path, header: dict) -> tuple:
    from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
    directory = Path('models/exchange_event_v2')
    models = FileExchangeModels.load(directory, lightweight=True)
    return tuple(cls(models, static_builder(record), _ExchangeEventEndSignals,
        FileM0Predictor(directory/'M0'), per_side_settled=header['per_side_settled'])
        for cls in (ExchangeEventOverlay, SplitExchangeOverlay))


def sample_pair(pair: tuple, smooth: tuple, context: dict, t: float) -> dict:
    from scripts.visualize_advantage_overlay import _exchange_display
    values = [_exchange_display(o, context.get('fallback_adv', 0.),
        context.get('fallback_p1', .5), s, t) for o, s in zip(pair, smooth)]
    raw = [o.tracker.probability for o in pair]
    return dict(t_sec=t, reference_raw=raw[0], candidate_raw=raw[1],
        raw_error=None if None in raw else abs(raw[0]-raw[1]),
        reference_probability=values[0][1], candidate_probability=values[1][1],
        probability_error=abs(values[0][1]-values[1][1]),
        advantage_error=abs(values[0][0]-values[1][0]),
        reference_source=pair[0].tracker.source, candidate_source=pair[1].tracker.source,
        landing_difference=([o._landing_projection.last for o in pair]
                            if raw[0] != raw[1] else None))


def summarize(rows: list[dict], frames: int, pair: tuple) -> dict:
    raw = [r['raw_error'] for r in rows if r['raw_error'] is not None]
    return dict(notifications=frames, publications=len(rows),
        raw_exact=sum(error == 0 for error in raw), raw_missing=len(rows)-len(raw),
        raw_probability_absolute_error=distribution(raw),
        displayed_probability_absolute_error=distribution([r['probability_error'] for r in rows]),
        displayed_advantage_absolute_error=distribution([r['advantage_error'] for r in rows]),
        source_mismatches=sum(r['reference_source'] != r['candidate_source'] for r in rows),
        physical_records_equal=physical_records(pair[0]) == physical_records(pair[1]),
        physical_record_count=len(pair[0].tracker.records), rows=rows)


def compare(record: Path, publication_times: set[float],
            calculation_times: set[float] | None = None) -> dict:
    from scripts.visualize_advantage_overlay import _ExchangeDisplayEMA, _exchange_display
    stream = read_records(record)
    pair = make_pair(record, next(stream))
    bridge = SimpleNamespace(notification_count=0)
    smooth = (_ExchangeDisplayEMA(), sampled_ema(_ExchangeDisplayEMA, bridge)())
    calculation_times = calculation_times if calculation_times is not None else publication_times
    frames, rows, context = 0, [], {}
    for item in stream:
        if item['kind'] != 'update':
            if item['kind'] == 'display':
                context = item
            continue
        inputs, t = item['args'], item['args'][3]
        for overlay in pair:
            overlay.update(*inputs)
        bridge.notification_count += 1
        frames += 1
        _exchange_display(pair[0], 0., .5, smooth[0], t)
        if t in calculation_times:
            pair[1].calculate()
            _exchange_display(pair[1], 0., .5, smooth[1], t)
        if t in publication_times:
            rows.append(sample_pair(pair, smooth, context, t))
    report = summarize(rows, frames, pair)
    report['requested_publications'] = len(publication_times)
    if len(rows) != len(publication_times):
        raise ValueError('公開時刻に対応する認識通知が欠落しています')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('logs/live_b6_measurement/cpu'))
    parser.add_argument('--output', type=Path, default=Path('logs/live_b7_comparison.json'))
    options = parser.parse_args()
    metrics = json.loads((options.input/'metrics.json').read_text())
    evaluations = json.loads((options.input/'evaluations.json').read_text())
    times = {r['frame']: r['t_sec'] for r in evaluations}
    sent = metrics['sse_sent']
    selected = {times[r['frame']] for r in sent if r.get('available', True)}
    calculations = metrics.get('probability_calculations')
    report = compare(options.input/'inputs.jsonl.gz', selected,
                     {r['t_sec'] for r in calculations} if calculations else None)
    if metrics.get('split_evaluation'):
        saved = {r['t_sec']: r for r in evaluations}
        report['candidate_replay_probability_absolute_error'] = distribution([
            abs(r['candidate_probability']-saved[r['t_sec']]['probability']) for r in report['rows']])
    report['sse_messages_including_repeats'] = len(sent)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
