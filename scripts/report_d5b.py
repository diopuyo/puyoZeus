"""D5bをD5と同じ門(logs/d5b/PREREGISTRATION.md)で採点し、q第14試合の誤確定0を必須にする。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import report_e35 as e35
from scripts import run_e17_ablation_20260928 as prior
from scripts.diagnose_d5 import directory
from scripts.report_e36b import SOURCE, game14_false_rows
from scripts.run_d5 import SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/d5b')
Q_MAX, ZENCHI_MIN, Q_FRAMES, ZENCHI_FRAMES = .509977, .9167, 6526, 8333
FIXED_KEYS = ('t_sec', 'game_idx', 'state1', 'state2', 'score1', 'score2')


def fixed_rows() -> dict:
    """E35と表示行の時刻・試合・状態・得点が一致することを確かめる。"""
    checks = {}
    for source in SOURCES:
        before = np.load(directory(Path('logs/e35/on'), source)/'display.npz')
        after = np.load(directory(OUT/'on', source)/'display.npz')
        for key in FIXED_KEYS:
            np.testing.assert_array_equal(before[key], after[key])
        checks[source] = len(after['t_sec'])
    return checks


def report() -> dict:
    """D5と同じ三条件+照合に、q第14試合の誤確定0と固定行一致を足す。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, Q_MAX, ZENCHI_MIN
    result = prior.report('on')
    assert result['q']['frames'] == Q_FRAMES and result['zenchi']['frames'] == ZENCHI_FRAMES
    e35.OUT = OUT
    audit = e35.audit_all(prior.report_base.death_metrics.deaths())
    result['gates']['false_certainty'] = audit['false'] == 0 and audit['unresolved'] == 0
    times = game14_false_rows(e35.events(SOURCE))
    result['game14_false_times'] = times
    result['gates']['game14_no_false'] = not times
    save_json(OUT/'FIXED_ROWS.json', fixed_rows())
    result['death_audit'] = {k: v for k, v in audit.items() if k != 'rows'}
    result['passed'] = all(result['gates'].values())
    save_json(OUT/'SUMMARY.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    report()
