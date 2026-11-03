"""独立OFF/ON再投入の判定・費用と、同一到着列でのキュー見積りを照合する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import bisect
import json
from pathlib import Path
import sys

sys.path.insert(0, _public_dev_text('<work>/wt_live_speed'))
from scripts.diagnose_live_speed3 import rows, stats
from scripts.summarize_live_speed3 import compare

OUT = Path(_public_dev_text('<work>/speed3'))
SOURCE = Path('C:/PuyoLive_20261012e/PuyoLive/output/spool')
FIX4 = Path(_public_dev_text('<work>/fix4'))
MS = 1000


def simulation(profiles: list, capture: dict, savings: dict) -> dict:
    """到着と元のアイドル隙間を固定。悪化した通知も差し引かず反映する。"""
    now, previous, delays, depths = 0., 0., [], []
    arrivals = [p['recognized_at'] for p in profiles]
    for index, profile in enumerate(profiles):
        idle = max(0., profile['started_at']-previous) if index else 0.
        if index == 0:
            now = profile['started_at']
        now = max(now+idle, profile['recognized_at']) + max(
            0., profile['evaluation_wall_sec']-savings.get(profile['frame'], 0.))
        previous = profile['finished_at']
        delays.append((now-capture[profile['frame']])*MS)
        depths.append(max(0, bisect.bisect_right(arrivals, now)-index-1))
    return dict(latency_ms=stats(delays), queue=stats(depths))


def timings(values: list, states: dict) -> dict:
    grouped: dict[str, list] = {k: [] for k in ('all', 'chain', 'ojama', 'other')}
    for row in values:
        group = states[row['t_sec']]
        cost = row['update_ms']+row['after_ms']
        grouped[group].append(cost)
        grouped['all'].append(cost)
    return {key: stats(value) for key, value in grouped.items()}


def queue_comparison(old: list, new: list) -> dict:
    profiles = list(rows(SOURCE/'profile.pickle'))
    recognized = list(rows(SOURCE/'recognition_rows.pickle'))
    frame = {r['t_sec']: r['frame'] for r in recognized}
    capture = {r['frame']: r['captured_at'] for r in recognized}
    reference = json.loads((FIX4/'reference_rows_pair.json').read_text(encoding='utf-8'))
    # 元e記録からFIX4の参照短縮分を引いてh相当を作る。未完了251件は差分0。
    base = {frame[r['t_sec']]: (r['before_ms']-r['after_ms'])/MS for r in reference}
    candidate = dict(base)
    for a, b in zip(old, new):
        key = frame[a['t_sec']]
        candidate[key] += (a['update_ms']+a['after_ms']-b['update_ms']-b['after_ms'])/MS
    return dict(note='保存到着・アイドル固定のh相当→候補見積り。実機計測ではない。',
                matched=len(old), profile_n=len(profiles),
                off=simulation(profiles, capture, base), on=simulation(profiles, capture, candidate))


def main() -> None:
    paths = [OUT/('candidate_'+mode) for mode in ('off', 'on')]
    values = [json.loads((p/'reference_rows_1.json').read_text(encoding='utf-8')) for p in paths]
    assert len(values[0]) == len(values[1])
    for a, b in zip(*values):
        assert all(a[k] == b[k] for k in ('t_sec', 'game', 'reference', 'display', 'ema'))
    signatures = [json.loads((p/'reference_summary_1.json').read_text())['digest'] for p in paths]
    assert signatures[0] == signatures[1]
    diagnostic = json.loads((OUT/'replay_diagnostic.json').read_text(encoding='utf-8'))
    states = {r['t_sec']: r['group'] for r in diagnostic['updates']}
    result = dict(updates=len(values[0]), digest=signatures[0],
                  invariance=[compare(p/'reference_replay_1') for p in paths],
                  timing_ms=dict(zip(('off', 'on'), [timings(v, states) for v in values])),
                  cache=json.loads((paths[1]/'cache.json').read_text()),
                  simulation=queue_comparison(*values))
    (OUT/'candidate_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
