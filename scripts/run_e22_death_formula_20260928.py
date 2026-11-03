"""E22: P99を固定してから、E19基準に窒息baseline限定ゲートを追加する。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.replay_exchange_event_20260926 import compare
from src.exchange_event_death_formula import FORMULA_HOLD_MAX_SEC

OUT = Path('logs/e22')
OPTIONS = dict(death_guard=True, confirmed_death_hold=True, death_formula_guard=True)
LOSS_MAX, AGREEMENT_MIN, SCENE_MIN = .513079, .8235, .85
SCENE = (2633.8, 2634.75)


class Trace(prior.AuditTrace):
    """保留の開始・確認・破棄を、勝敗指標と独立に記録する。"""
    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.guard: Any = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        self.guard = overlay._formula_guard


def worker(source: str, off: bool) -> None:
    """モデル・元入力・基準フラグをE19と同じに固定する。"""
    prior.OUT = OUT
    trace = Trace(source)
    prior.AuditTrace = lambda _: trace
    variant = 'off' if off else 'on'
    options = dict(death_guard=True, confirmed_death_hold=True) if off else OPTIONS
    prior.locked_worker(variant, source, options)
    dest = prior.directory(variant, source)
    if trace.guard is not None:
        save_json(dest/'formula_guard.json', trace.guard.audit)
    if off:
        save_json(OUT/'OFF_EQUIVALENCE.json', compare(Path('logs/e19/hold/review'), dest))


def launch(source: str) -> None:
    """2並列で固定再生し、完了単位から再開できるようにする。"""
    dest = prior.directory('on', source)
    if (dest/'DONE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'scripts.run_e22_death_formula_20260928',
                        '--source', source], stdout=log, stderr=subprocess.STDOUT, check=True)


def hold_summary() -> dict:
    """保留件数は同じbaseline発火時刻を一件とし、欠測を成功に数えない。"""
    sources, total = {}, Counter()
    for source in prior.ALL_SOURCES:
        rows = json.loads((prior.directory('on', source)/'formula_guard.json').read_text())
        counts = Counter(r['outcome'] for r in rows)
        summary = dict(held=len(rows), confirmed=sum(n for k, n in counts.items() if k.startswith('confirmed')),
                       discarded=sum(n for k, n in counts.items() if k.startswith('discarded')),
                       unresolved=counts['pending'])
        sources[source] = dict(summary=summary, outcomes=dict(counts), rows=rows)
        if source in SOURCES:
            total.update(summary)
    return dict(three_videos=dict(total), sources=sources)


def report() -> dict:
    """事前登録した四つのゲートを未丸め値で判定する。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    value = prior.report('on')
    data = np.load(prior.directory('on', 'review')/'display.npz')
    mask = (data['t_sec'] >= SCENE[0]) & (data['t_sec'] <= SCENE[1])
    value['scene'] = dict(frames=int(mask.sum()), minimum=float(data['display_p1'][mask].min()))
    value['gates']['scene'] = value['scene']['minimum'] >= SCENE_MIN
    value['candidate'] = all(value['gates'].values())
    save_json(OUT/'on/METRICS.json', value)
    save_json(OUT/'HOLDS.json', hold_summary())
    return value


def main() -> None:
    """時間上限の由来と採否を、評価に先立って登録する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    if args.source:
        worker(args.source, args.off)
        return
    if not args.report_only:
        calibration = json.loads((OUT/'FORMULA_DELAY.json').read_text())
        assert calibration['max_hold_sec'] == FORMULA_HOLD_MAX_SEC
        save_json(OUT/'PROTOCOL.json', dict(options=OPTIONS, baseline='E19 / ba614f7',
            q_max=LOSS_MAX, zenchi_min=AGREEMENT_MIN, false_max=1, reference_firings=28,
            scene=SCENE, scene_min=SCENE_MIN, max_hold_sec=FORMULA_HOLD_MAX_SEC,
            calibration=calibration, score_evidence='会計前の表示得点が発火直前より増加',
            timeout='上限内に確認できないbaselineは破棄。式由来の通知自体は変更しない。'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
