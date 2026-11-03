"""短区間と全体の同じ入力部分、および判定を原票から再確認する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2_ladder_20261002 import STOP

DEST = OUT/'v2d'
TIME_POINT_MAX, TIME_CI_MAX = .002, .020
AGREE_POINT_MIN, AGREE_CI_MIN = -.005, -.030
SCENE_DEADLINE, REVERSAL_RATIO = 2766., 1.2


def prefix(source: str, end: float) -> dict:
    """実処理時間だけを分離し、同じ入力部分の表示・H1・E35判定を照合する。"""
    short, full = DEST/'short'/source, OUT/'replay/v2d'/source
    with np.load(short/'display.npz') as left, np.load(full/'display.npz') as right:
        mask = right['t_sec'] <= end
        for key in ('t_sec','game_idx','display_p1','display_adv','source'):
            np.testing.assert_array_equal(left[key], right[key][mask])
    elapsed_differences = 0
    for filename in ('history_integrity_audit.json','post_counter_bound_audit.json'):
        left = json.loads((short/filename).read_text())['rows']
        right = [r for r in json.loads((full/filename).read_text())['rows'] if r['t_sec'] <= end]
        if filename == 'post_counter_bound_audit.json':
            # 証明返却値とは別のperf_counter測定値だけを除外する（原票は保持）。
            assert len(left) == len(right)
            elapsed_differences = sum(a['elapsed_sec'] != b['elapsed_sec'] for a,b in zip(left,right))
            left = [{k:v for k,v in r.items() if k != 'elapsed_sec'} for r in left]
            right = [{k:v for k,v in r.items() if k != 'elapsed_sec'} for r in right]
        assert left == right, (source,filename)
    return dict(display='identical', history_audit='identical',
                post_counter_audit='identical_except_elapsed_sec', elapsed_differences=elapsed_differences)


def gates() -> dict:
    """閾値を原票の点推定・区間から読み直し、採点器の真偽と一致を確認する。"""
    result = json.loads((DEST/'RESULT_v2d.json').read_text())
    time = result['combined']['metrics']['m_time']['paired']
    agree = result['combined']['metrics']['m_agree']['paired']
    checks = dict(m_time=time['diff'] <= TIME_POINT_MAX and time['ci95'][1] < TIME_CI_MAX,
        m_agree=agree['diff'] >= AGREE_POINT_MIN and agree['ci95'][0] > AGREE_CI_MIN,
        false_death=result['existing5']['false']==result['existing5']['unresolved']==0 and all(
            r['deaths']['after']['false']==r['deaths']['after']['false_frames']==0 for r in result['sets'].values()),
        regression=result['regression']['scene']['legacy_metric_sec'] <= SCENE_DEADLINE and not result['regression']['game14_false_times'],
        reversals=all(r['v2d'] <= REVERSAL_RATIO*r['prod'] for r in result['reversals'].values()))
    assert checks == result['gates_v2']
    return checks


def main() -> None:
    """独立採否ではなく、同一入力と集計値の内部検算として記録する。"""
    checks = {source:prefix(source,end) for source,end in STOP.items()}
    files = [Path(p) for p in ('src/exchange_history_integrity.py','src/exchange_event_overlay.py',
        'src/exchange_event_landing.py','src/exchange_event_tracker.py',
        'scripts/replay_exchange_event_20260926.py','scripts/falsedeath_replay_20261001.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with (DEST/'VERIFY.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(prefix=checks, gates=gates(), source_sha256=hashes,
            note='統括の査読・独立検収の代替ではない'), stream, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
