"""単独比較で選んだ要素と途中完走予測を固定記録で検収する。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts import run_e17_ablation_20260928 as prior
from scripts import run_e23_multilanding_20260928 as e23
from scripts import audit_e23_deaths_20260928 as deaths
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e26')
WORKERS = 2


def worker(source: str, off: bool = False) -> None:
    """判定ラベル・モデルは固定し、追加観測付き記録だけを入力する。"""
    prior.OUT, prior.AuditTrace = OUT, e23.ProgressTrace
    options = json.loads((OUT/'ABLATION.json').read_text())['options']
    if off:
        options = dict(death_guard=True, confirmed_death_hold=True, death_formula_guard=True)
    else:
        options = dict(options, midchain_completion=True)
    def enriched(record: Path, *args: object, **kwargs: object) -> dict:
        """元の再生器へ予測専用の追加入力を渡す。"""
        return replay(OUT/'records'/record.name, *args, **kwargs)
    prior.replay = enriched
    prior.locked_worker('off' if off else 'on', source, options)
    if off:
        before = Path('logs/e22/on') / (source if source in ('zenchi', 'review') else f'renders/{source}/on')
        save_json(OUT/f'OFF_{source}.json', compare(before, prior.directory('off', source)))


def launch(source: str) -> None:
    """最大二プロセスで再生し、異常終了を個別ログに残す。"""
    dest = prior.directory('on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e26_completion', '--source', source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """採否を固定閾値で判定し、採用後の未確認も独立に数える。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, .513079, .8262
    value = prior.report('on')
    stamp = first(*displayed(prior.directory('on', 'review')/'display.npz'))
    value['scene'] = dict(first_sec=stamp, deadline_sec=2766., scenes=1,
        baseline_first_sec=first(*displayed(Path('logs/e22/on/review/display.npz'))))
    value['gates']['scene'] = stamp is not None and stamp <= 2766.
    summaries, combined = {}, Counter()
    for source in prior.ALL_SOURCES:
        summary = json.loads((prior.directory('on', source)/'midchain_audit.json').read_text())
        summaries[source] = {k: v for k, v in summary.items() if k not in ('rows', 'skipped')}
        if source in SOURCES:
            combined.update(summaries[source])
    save_json(OUT/'MIDCHAIN_COUNTS.json', dict(three_videos=dict(combined), sources=summaries))
    value['midchain_three_videos'] = dict(combined)
    value['candidate'] = all(value['gates'].values())
    e23.OUT, deaths.OUT = OUT, OUT
    save_json(OUT/'DEATH_AUDIT.json', e23.death_audit())
    deaths.main()
    save_json(OUT/'on/METRICS.json', value)
    return value


def main() -> None:
    """単独比較が完了するまでは組合せを起動できないようにする。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--off', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    assert (OUT/'ABLATION.json').exists(), '先に単独切り分けを完了する'
    if args.source:
        worker(args.source, args.off)
        return
    if not args.report_only:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
