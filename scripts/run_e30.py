"""E27条件に発火候補復元だけを追加し、事前登録条件で固定再生する。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e27 import OPTIONS as E27_OPTIONS
from scripts.audit_e30_predictions import CandidateTrace
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e30')
OPTIONS = dict(E27_OPTIONS, prefire_candidates=True)
WORKERS = 2
LOSS_MAX, AGREEMENT_MIN, DEADLINE = .512778, .8468, 2766.
PERCENTILES = (50, 95)


def worker(source: str, control: bool = False) -> None:
    """認識・モデル・入力を変更せず、OFFはE27と全出力を照合する。"""
    prior.OUT, prior.AuditTrace = OUT, CandidateTrace
    variant, options = ('off', E27_OPTIONS) if control else ('on', OPTIONS)
    def enriched(record: Path, dest: Path, *args: object, **kwargs: object) -> dict:
        """同一の補完済み記録を読み、連鎖参照を使う事後監査だけを保存する。"""
        result = replay(Path('logs/e26/records')/record.name, dest, *args, **kwargs)
        if not control:
            save_json(dest/'prefire_final_audit.json', args[2].summary())
        return result
    prior.replay = enriched
    prior.locked_worker(variant, source, options)
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', compare(Path('logs/e27/on')/suffix, prior.directory(variant, source)))


def launch(task: tuple[str, bool]) -> None:
    """最大二再生。完了済みの同一条件だけを再利用する。"""
    source, control = task
    dest = prior.directory('off' if control else 'on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        command = [sys.executable, '-m', 'scripts.run_e30', '--source', source]
        subprocess.run(command+(['--control'] if control else []),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def quantiles(values: list) -> dict:
    """未確定値を除き、P50/P95と実際の母数を一緒に保存する。"""
    known = [v for v in values if v is not None]
    return dict(n=len(known), p50=float(np.percentile(known, PERCENTILES[0])) if known else None,
                p95=float(np.percentile(known, PERCENTILES[1])) if known else None)


def counts(rows: list[dict]) -> dict:
    """列挙不能と観測で0になった発火を分け、予測時点ごとのずれを残す。"""
    eligible = [r for r in rows if r['skipped'] is None]
    remaining = [r for r in eligible if r['remaining']]
    errors = lambda key, selected: quantiles([abs(r[key]) if r[key] is not None else None for r in selected])
    return dict(fires=len(rows), enumerated=len(eligible), skipped=len(rows)-len(eligible),
        remaining=len(remaining), zero=sum(not r['remaining'] for r in eligible),
        became_zero=sum(r['became_zero'] for r in eligible),
        final_unresolved=sum(r['final_score'] is None for r in remaining),
        remaining_last_absolute_error=errors('last_error', remaining),
        first_absolute_error=errors('first_error', eligible), last_absolute_error=errors('last_error', eligible),
        enumeration_seconds=quantiles([r['enumeration_sec'] for r in eligible]),
        filtering_seconds=quantiles([o['filter_sec'] for r in eligible for o in r['observations']]))


def report() -> dict:
    """閾値を変更せず全条件を判定し、3動画と重複するレビューを集計から分ける。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    value = prior.report('on')
    save_json(OUT/'DEATH_AUDIT.json', prior.report_base.death_metrics.deaths())
    deaths = value['deaths']
    value['gates']['deaths'] = deaths['unlabelled'] == 0 and deaths['false']/max(1, deaths['total']) <= 1/prior.DEATHS
    stamp = first(*displayed(prior.directory('on', 'review')/'display.npz'))
    value['scene'] = dict(first_sec=stamp, deadline_sec=DEADLINE, scenes=1, baseline_first_sec=2771.25)
    value['gates']['scene'] = stamp is not None and stamp <= DEADLINE
    rows = {s: json.loads((prior.directory('on', s)/'prefire_final_audit.json').read_text())['rows']
            for s in prior.ALL_SOURCES}
    counts_value = dict(three_videos=counts([r for s in SOURCES for r in rows[s]]),
                       sources={s: counts(r) for s, r in rows.items()})
    save_json(OUT/'CANDIDATE_COUNTS.json', counts_value)
    save_json(OUT/'SCENE_CANDIDATES.json', [r for r in rows['review']
        if r['side'] == '2P' and 2750 <= r['trigger_sec'] <= 2755])
    value['candidate'] = all(value['gates'].values())
    save_json(OUT/'on/METRICS.json', value)
    return value


def main() -> None:
    """事前登録後にON検収とOFF対照を実行する。"""
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
        tasks = [(s, False) for s in ('review', *SOURCES, 'zenchi')]+[('review', True), (SOURCES[0], True)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, tasks))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
