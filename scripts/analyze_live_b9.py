"""B9の5分推移・17試合突合・故障時のSSEゲートを集計する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from scripts.measure_live_b6 import save

WINDOW_SEC = 300.
FPS = 30.
EXPECTED_GAMES = 17
TIME_RESOLUTION_SEC = .001


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    complete = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    return [json.loads(line) for line in complete if line.strip()]


def quantile(values: list[float], percentile: int = 95) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def available(row: dict) -> bool:
    return row['payload'].get('evaluations', {}).get('practical', {}).get('availability') == 'available'


def input_hold(row: dict | None) -> bool:
    return row is not None and not available(row) and row['payload'].get('display', {}).get('status') == 'hold'


def samples(path: Path) -> list[dict]:
    """heartbeat等を除外し、実際に購読したDTOだけを使う。"""
    seen, output = set(), []
    for row in lines(path/'sse.jsonl'):
        identity = row['payload'].get('identity')
        if identity and identity['stream_seq'] not in seen:
            seen.add(identity['stream_seq'])
            output.append(row)
    return output


def windows(path: Path, start: float, end: float) -> list[dict]:
    expected_total = int(json.loads((path/'metrics.json').read_text())['expected_frames'])
    runtime = lines(path/'runtime.jsonl')
    resources, sent = lines(path/'resources.jsonl'), samples(path)
    drops = json.loads((path/'recognition.json').read_text())['dropped_times']
    output = []
    for value in np.arange(start, end, WINDOW_SEC):
        # NumPyの比較結果をsumへ渡すとint64へ伝播し、JSON保存に失敗する。
        lo = float(value)
        hi = float(min(lo+WINDOW_SEC, end))
        ticks = [r for r in runtime if lo <= r['progress'].get('t_sec', -1) < hi]
        selected = [r for r in resources if ticks and ticks[0]['at'] <= r['at'] <= ticks[-1]['at']]
        rows = [r for r in sent if available(r) and
                lo <= r['payload']['timing']['source_available_ms']/1000 < hi]
        delay = [(r['received_at']-r['payload']['timing']['capture_monotonic_sec'])*1000 for r in rows]
        expected = round((hi-lo)*FPS)
        if hi == end:
            expected = expected_total-sum(row['expected'] for row in output)
        drop = sum(lo <= t < hi for t in drops)
        output.append(dict(start_sec=float(lo), end_sec=float(hi), resource_samples=len(selected),
            rss_p50=quantile([r['rss_bytes'] for r in selected], 50),
            handles_p50=quantile([r['handles'] for r in selected], 50),
            vram_p50=quantile([r['vram_mib'] for r in selected if r['vram_mib'] is not None], 50),
            queue_p95=quantile([r['queue_depth'] for r in ticks]),
            queue_max=max([r['queue_depth'] for r in ticks], default=None),
            dropped=drop, expected=expected, drop_fraction=drop/expected,
            latency_p95_ms=quantile(delay), latency_samples=len(delay)))
    return output


def trends(rows: list[dict]) -> dict:
    output = {}
    for key in ('rss_p50', 'handles_p50', 'vram_p50', 'queue_p95', 'drop_fraction', 'latency_p95_ms'):
        values = [row[key] for row in rows if row[key] is not None]
        output[key] = dict(values=values, delta=values[-1]-values[0] if len(values) > 1 else None,
            strictly_increasing=len(values) >= 3 and all(b > a for a, b in zip(values, values[1:])),
            interpretation='上昇は要調査。3区間だけでリーク有無を断定しない')
    return output


def games(path: Path, start: float, end: float) -> dict:
    runtime = lines(path/'runtime.jsonl')
    starts = runtime[-1]['game_starts'] if runtime else []
    starts = [r for r in starts if start <= r['t_sec'] < end]
    sent = samples(path)
    rows = []
    for i, game in enumerate(starts):
        limit = starts[i+1]['t_sec'] if i+1 < len(starts) else end
        first = next((r for r in sent if available(r) and
            r['payload']['identity']['match_id'] == str(game['game']) and
            game['t_sec'] <= r['payload']['timing']['source_available_ms']/1000 < limit), None)
        rows.append(dict(**game, first_probability_delay_sec=None if first is None else
            first['received_at']-game['captured_at'], first_probability=None if first is None else
            first['payload']['evaluations']['practical']['p1_win_probability']))
    unique = len({row['game'] for row in rows})
    return dict(expected=EXPECTED_GAMES, detected=len(rows), unique_game_ids=unique,
        expected_internal_boundaries=EXPECTED_GAMES-1, detected_internal_boundaries=max(0, unique-1),
        count_matches=len(rows) == unique == EXPECTED_GAMES,
        rows=rows, note='開始は認識の試合内遷移。17件一致は個々の手動境界時刻との一致を保証しない')


def fault_result(path: Path) -> dict:
    events, sent = lines(path/'faults.jsonl'), samples(path)
    starts = [r for r in events if r['phase'] == 'start']
    ends = [r for r in events if r['phase'] == 'end']
    if len(starts) != 1 or len(ends) != 1:
        return dict(passed=False, reason='故障開始/終了記録が一件ずつ揃っていない', events=events)
    begin, end = starts[0], ends[0]
    during = [r for r in sent if begin['at'] <= r['received_at'] < end['at']]
    after = [r for r in sent if r['received_at'] >= end['at']]
    first = after[0] if after else None
    restored = next((r for r in after if available(r)), None)
    unsafe = [r for r in sent if r['payload'].get('display', {}).get('input_status') in
              ('verifying', 'calibrating', 'no_puyo_screen') and available(r)]
    phases = [r['payload'].get('display', {}).get('input_status') for r in during+after]
    stale = [r for r in after if available(r) and
             r['payload']['timing']['source_available_ms']/1000+TIME_RESOLUTION_SEC < end['t_sec']]
    invalid_frames = [r for r in sent if available(r) and
        begin['t_sec'] <= r['payload']['timing']['source_available_ms']/1000 < end['t_sec']]
    required = {'calibrating', 'ready', 'no_puyo_screen' if begin['kind'] in ('black', 'other') else 'verifying'}
    missing = sorted(required-set(phases))
    exposed = sum(available(r) for r in during)
    return dict(kind=begin['kind'], passed=restored is not None and not unsafe and not stale and not exposed
                and not invalid_frames and not missing and input_hold(first),
        recovery_sec=None if restored is None else restored['received_at']-end['at'],
        first_after_recovery=None if first is None else first['payload'],
        first_is_hold=input_hold(first),
        input_statuses=list(dict.fromkeys(phases)), invalid_status_probability_count=len(unsafe),
        stale_after_recovery_count=len(stale), available_during_fault=exposed,
        invalid_frame_probability_count=len(invalid_frames), missing_statuses=missing,
        note='数値のGTがないため最初の復帰表示はHOLDを要求。数値なら正しいと推測せず未合格とする')


def report(path: Path, start: float, end: float, fault: bool = False) -> dict:
    if fault:
        result = fault_result(path)
    else:
        timeline = windows(path, start, end)
        result = dict(windows=timeline, trends=trends(timeline), games=games(path, start, end))
    save(path/'acceptance.json', result)
    return result
