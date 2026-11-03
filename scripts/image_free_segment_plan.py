"""実測試合境界から所有区間を作り、処理能力に比例して二台へ割り当てる。"""
from __future__ import annotations

from typing import Iterable

from scripts.image_free_segment_merge import validate_segments

OWNED_LEAD_SECONDS = 1.0
WARMUP_SECONDS = 5.0
MIN_SEGMENT_SECONDS = 15.0


def partition(video: dict, boundaries: Iterable[float], fps: float, count: int) -> list[dict]:
    cuts = [0]
    for seconds in sorted(set(boundaries)):
        cut = max(0, round((seconds-OWNED_LEAD_SECONDS)*fps))
        if cut-cuts[-1] >= MIN_SEGMENT_SECONDS*fps and count-cut >= MIN_SEGMENT_SECONDS*fps:
            cuts.append(cut)
    cuts.append(count)
    segments = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        segments.append(dict(video, id=f"{video['id']}_s{index:04d}", video_id=video['id'],
            start_frame=max(0, start-round(WARMUP_SECONDS*fps)), owned_start=start,
            end_frame=end, fps=fps, interval_evidence=video['interval_evidence']))
    return segments


def assign(segments: list[dict], rates: dict[str, float]) -> list[dict]:
    """助走も処理費用に含め、残り終了時刻が最短になる台へ長い区間から置く。"""
    if set(rates) != {'pc1', 'pc2'} or any(r <= 0 for r in rates.values()):
        raise ValueError('両PCの正の合計フレーム/秒が必要です')
    seconds = dict.fromkeys(rates, 0.)
    assigned = {}
    for segment in sorted(segments, key=lambda s: s['end_frame']-s['start_frame'], reverse=True):
        cost = segment['end_frame']-segment['start_frame']
        host = min(rates, key=lambda pc: seconds[pc]+cost/rates[pc])
        seconds[host] += cost/rates[host]
        assigned[segment['id']] = dict(segment, host=host)
    return [assigned[s['id']] for s in segments]


def plan(videos: list[dict], observations: dict, rates: dict[str, float]) -> dict:
    segments, expected = [], {}
    for video in videos:
        source = observations[video['id']]
        expected[video['id']] = source['frames']
        segments.extend(partition(video, source['boundaries'], source['fps'], source['frames']))
    validate_segments(segments, expected)
    return dict(segments=assign(segments, rates), expected_frames=expected,
                rates=rates, owned_lead_seconds=OWNED_LEAD_SECONDS, warmup_seconds=WARMUP_SECONDS)
