"""既存の表示更新から、個人情報を持たない小さな数値要約を作る。"""
from __future__ import annotations

from collections import deque
import math
import statistics
import time
from typing import Any

MAX_MATCHES = 20
MAX_POINTS = 24
MAX_DELAYS = 2048
SAMPLE_SECONDS = 1.0
ROUND_DIGITS = 4
DELAY_PERCENTILE = 0.9
DOWNSAMPLE_FACTOR = 2


def number(value: Any) -> float | None:
    """bool、文字列、非有限値を数値として送らない。"""
    if type(value) not in (int, float):
        return None
    return float(value) if math.isfinite(value) else None


class NumericSummary:
    """更新通知だけを観測する。試合IDは比較のみに用い、出力には連番を使う。"""

    def __init__(self) -> None:
        self.observations = 0
        self.recognition_observations = 0
        self.recognition_unreliable = 0
        self.match_count = 0
        self.last_match: Any = None
        self.matches: deque[dict[str, Any]] = deque(maxlen=MAX_MATCHES)
        self.delays: deque[float] = deque(maxlen=MAX_DELAYS)
        self.delay_observations = 0
        self.last_sample = -math.inf

    def observe(self, data: dict[str, Any]) -> None:
        self.observations += 1
        display = data.get('display', {})
        if 'primary_hold_reason' in display:
            self.recognition_observations += 1
            self.recognition_unreliable += int(display['primary_hold_reason'] == 'recognition_unreliable')
        delay = number(data.get('timing', {}).get('capture_to_publish_latency_ms'))
        if delay is not None and delay >= 0:
            self.delays.append(round(delay, ROUND_DIGITS))
            self.delay_observations += 1
        self.observe_match(data)

    def observe_match(self, data: dict[str, Any]) -> None:
        identity, display = data.get('identity', {}), data.get('display', {})
        match_id = identity.get('match_id')
        if match_id is None or display.get('match_phase') not in ('playing', 'terminal'):
            return
        match = (identity.get('stream_id'), match_id)
        if match != self.last_match:
            self.last_match = match
            self.match_count += 1
            self.matches.append(dict(index=self.match_count, observations=0, stride=1, points=[]))
            self.last_sample = -math.inf
        probability = number(data.get('evaluations', {}).get('practical', {}).get('p1_win_probability'))
        if probability is None or not 0 <= probability <= 1:
            return
        now = time.monotonic()
        if now - self.last_sample < SAMPLE_SECONDS:
            return
        self.last_sample = now
        current = self.matches[-1]
        current['observations'] += 1
        if (current['observations'] - 1) % current['stride']:
            return
        current['points'].append(round(probability, ROUND_DIGITS))
        if len(current['points']) > MAX_POINTS:
            current['points'] = current['points'][::DOWNSAMPLE_FACTOR]
            current['stride'] *= DOWNSAMPLE_FACTOR

    def snapshot(self, uptime_seconds: float) -> dict[str, Any]:
        """分母と保持範囲を併記する。欠測はnullで表す。"""
        ordered = sorted(self.delays)
        median = statistics.median(ordered) if ordered else None
        p90 = ordered[math.ceil(len(ordered) * DELAY_PERCENTILE) - 1] if ordered else None
        denominator = self.recognition_observations
        return dict(uptime_seconds=round(max(0, uptime_seconds), ROUND_DIGITS),
                    observations=self.observations, match_count=self.match_count,
                    retained_matches=len(self.matches),
                    matches=[dict(index=m['index'], sampled_observations=m['observations'],
                                  stride=m['stride'], p1_win_probabilities=list(m['points']))
                             for m in self.matches],
                    delay_observations=self.delay_observations, delay_window_samples=len(ordered),
                    delay_median_ms=median, delay_p90_ms=p90,
                    recognition_observations=denominator,
                    recognition_unreliable_observations=self.recognition_unreliable,
                    recognition_unreliable_fraction=(round(self.recognition_unreliable / denominator,
                                                          ROUND_DIGITS) if denominator else None))
