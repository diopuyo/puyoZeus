"""E31の全発火監査と事前登録条件の固定再生。"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e23_multilanding_20260928 import ProgressTrace
from scripts.run_e27 import OPTIONS as BASE
from scripts.run_e30 import quantiles
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path('logs/e31')
OPTIONS = dict(BASE, prefire_snapshot=True)
WORKERS = 2
RECORD_WAIT_SECONDS = 5


class SnapshotTrace(ProgressTrace):
    """予測時点の値は変更せず、確定した実最終得点を事後監査する。"""
    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.chains: dict = {}
        self.engine: Any = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        self.engine = overlay._prefire
        if self.engine is not None:
            self.chains.update({(e['audit']['game'], e['chain'].side, key): e['chain']
                for key,e in self.engine.entries.items()})

    def summary(self) -> dict:
        """途中撤回を含む全採用の初回予測誤差を残す。"""
        rows = self.engine.summary()['rows']
        for row in rows:
            chain = self.chains[(row['game'], row['side'], row['chain_id'])]
            ready = chain.score_ready_sec is not None and chain.end_signal_sec is not None and chain.end_confirmed is not False
            row['final_score'] = chain.score_delta if ready else None
            row['error'] = row['predicted_score']-chain.score_delta if row['accepted'] and ready and chain.score_delta is not None else None
        return dict(rows=rows)


def worker(source: str, control: bool) -> None:
    """OFFも追加観測付き入力で再生し、E27保存出力と全列照合する。"""
    prior.OUT, prior.AuditTrace = OUT, SnapshotTrace
    def enriched(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        value = replay(OUT/'records'/record.name, dest, *args, **kwargs)
        if not control:
            save_json(dest/'snapshot_final_audit.json', args[2].summary())
        return value
    prior.replay = enriched
    prior.locked_worker('off' if control else 'on', source, BASE if control else OPTIONS)
    if control:
        suffix = source if source in ('review','zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', compare(Path('logs/e27/on')/suffix, prior.directory('off',source)))


def launch(task: tuple[str, bool]) -> None:
    """固定入力の再生を最大二プロセスで実行する。"""
    source, control = task
    while not (OUT/'records'/f'{source}.jsonl.json').exists():
        time.sleep(RECORD_WAIT_SECONDS)
    dest = prior.directory('off' if control else 'on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as log:
        subprocess.run([sys.executable, '-m', 'scripts.run_e31', '--source', source]+
            (['--control'] if control else []), stdout=log, stderr=subprocess.STDOUT, check=True)


def counts(rows: list[dict]) -> dict:
    """全発火を分母にし、不採用理由と撤回を排他的な終状態として集計する。"""
    accepted = [r for r in rows if r['accepted']]
    errors = [abs(r['error']) for r in accepted if r['error'] is not None]
    rejected = Counter(r['reason'] for r in rows if not r['accepted'])
    checks = dict(window=sum(rejected[k] for k in ('no_landed_window','missing_snapshot','invalid_shape')),
        a=sum(rejected[k] for k in ('floating','missing_origin','origin_difference')),
        b=sum(rejected[k] for k in ('unknown','palette')),
        c=sum(rejected[k] for k in ('missing_first_formula','first_score')))
    return dict(fires=len(rows), accepted=len(accepted), retained=sum(not r['withdrawn'] for r in accepted),
        withdrawn=sum(r['withdrawn'] for r in accepted),
        rejected=dict(rejected), rejected_checks=checks,
        score_absolute_error=quantiles(errors), exact=sum(e == 0 for e in errors),
        unresolved=len(accepted)-len(errors))


def report() -> dict:
    """同じ行と固定閾値で比較し、対象場面も必須ゲートに含める。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, .512778, .8468
    value = prior.report('on')
    deaths = value['deaths']
    value['gates']['deaths'] = deaths['unlabelled'] == 0 and deaths['false']/max(1,deaths['total']) <= 1/28
    stamp = first(*displayed(prior.directory('on','review')/'display.npz'))
    value['scene'] = dict(first_sec=stamp, deadline_sec=2766., scenes=1)
    value['gates']['scene'] = stamp is not None and stamp <= 2766.
    rows = {s:json.loads((prior.directory('on',s)/'snapshot_final_audit.json').read_text())['rows'] for s in prior.ALL_SOURCES}
    save_json(OUT/'SNAPSHOT_COUNTS.json', dict(three_videos=counts([r for s in SOURCES for r in rows[s]]),
        sources={s:counts(r) for s,r in rows.items()}))
    save_json(OUT/'SCENE.json', [r for r in rows['review'] if r['side']=='2P' and 2750<=r['trigger_sec']<=2755])
    value['candidate'] = all(value['gates'].values())
    save_json(OUT/'on/METRICS.json', value)
    save_json(OUT/'DEATH_AUDIT.json', prior.report_base.death_metrics.deaths())
    return value


def main() -> None:
    """入力補完完了後にON/OFFを全動画で検収する。"""
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
            list(pool.map(launch, [(s,c) for c in (False,True) for s in ('review',*SOURCES,'zenchi')]))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
