"""死亡専用の台帳・隠し段上限を固定入力で検収する。"""
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

OUT = Path('logs/e27')
OPTIONS = dict(death_guard=True, confirmed_death_hold=True, death_formula_guard=True,
    color_score_safety=True, multi_landing_death=True, midchain_completion=True,
    death_pending_ledger=True, hidden_row_death=True)
WORKERS = 2


def worker(source: str, control: bool = False) -> None:
    """入力とモデルを固定し、制御条件はE26出力とバイト照合する。"""
    prior.OUT, prior.AuditTrace = OUT, e23.ProgressTrace
    options = {k: v for k, v in OPTIONS.items() if not control or k not in ('death_pending_ledger', 'hidden_row_death')}
    def enriched(record: Path, *args: object, **kwargs: object) -> dict:
        """途中観測を補完済みの同一記録を使う。"""
        return replay(Path('logs/e26/records')/record.name, *args, **kwargs)
    prior.replay = enriched
    prior.locked_worker('control' if control else 'on', source, options)
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'CONTROL_{source}.json', compare(Path('logs/e26/on')/suffix, prior.directory('control', source)))


def launch(source: str) -> None:
    """最大二プロセスで検収する。"""
    dest = prior.directory('on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e27', '--source', source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """結果によって閾値や採用条件を変更しない。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, .513079, .8262
    value = prior.report('on')
    stamp = first(*displayed(prior.directory('on', 'review')/'display.npz'))
    value['scene'] = dict(first_sec=stamp, deadline_sec=2766., scenes=1)
    value['gates']['scene'] = stamp is not None and stamp <= 2766.
    summaries, combined = {}, Counter()
    for source in prior.ALL_SOURCES:
        audit = json.loads((prior.directory('on', source)/'hidden_death_audit.json').read_text())
        summaries[source] = {k: v for k, v in audit.items() if k not in ('rows', 'skipped')}
        if source in SOURCES:
            combined.update(summaries[source])
    save_json(OUT/'HIDDEN_COUNTS.json', dict(three_videos=dict(combined), sources=summaries))
    value['hidden_three_videos'] = dict(combined)
    value['candidate'] = all(value['gates'].values())
    e23.OUT, deaths.OUT = OUT, OUT
    save_json(OUT/'DEATH_AUDIT.json', e23.death_audit())
    deaths.main()
    save_json(OUT/'on/METRICS.json', value)
    return value


def main() -> None:
    """事前登録が存在する状態でだけ固定条件を実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    prior.OUT = OUT
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source, args.control)
        return
    if not args.report_only:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
