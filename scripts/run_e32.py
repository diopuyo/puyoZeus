"""E32の固定入力ON/OFF検収。閾値は実行前に固定する。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any
from scripts import run_e31 as e31
from scripts.run_e3_exchange_eval_20260926 import save_json, SOURCES
from scripts.replay_exchange_event_20260926 import replay, compare
from src.exchange_event_overlay import ExchangeEventOverlay
from src.hidden_row_belief import CALIBRATION_BINS

OUT = Path('logs/e32')
OPTIONS = dict(e31.OPTIONS, hidden_row_belief=True)
WORKERS = 2


class BeliefTrace(e31.SnapshotTrace):
    """発火と自己較正の原票を同じ動画単位で保存する。"""
    def summary(self) -> dict:
        value = super().summary()
        value['calibration'] = self.engine.calibration
        return value


def worker(source: str, control: bool) -> None:
    """OFFはE27条件、ONは二フラグ追加だけで同じ全行を再生する。"""
    prior = e31.prior
    prior.OUT, prior.AuditTrace = OUT, BeliefTrace
    timings = []
    update = ExchangeEventOverlay.update
    def timed(overlay: Any, *args: Any, **kwargs: Any) -> None:
        """入力復号や監査出力を含めず評価更新の所要時間を測る。"""
        started = perf_counter()
        update(overlay, *args, **kwargs)
        timings.append(perf_counter()-started)
    ExchangeEventOverlay.update = timed
    def enriched(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        """E31と全く同じ追加観測付き原票を読む。"""
        value = replay(Path('logs/e31/records')/record.name, dest, *args, **kwargs)
        if not control:
            save_json(dest/'snapshot_final_audit.json', args[2].summary())
        save_json(dest/'TIMING.json', dict(update_sec=e31.quantiles(timings), total_sec=sum(timings)))
        return value
    prior.replay = enriched
    prior.locked_worker('off' if control else 'on', source, e31.BASE if control else OPTIONS)
    if control:
        suffix = source if source in ('review','zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', compare(Path('logs/e27/on')/suffix, prior.directory('off',source)))


def launch(task: tuple) -> None:
    """再開可能な動画単位で最大二プロセスを起動する。"""
    source, control = task
    e31.prior.OUT = OUT
    dest = e31.prior.directory('off' if control else 'on', source)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'runner.log').open('a') as stream:
        subprocess.run([sys.executable, '-m', 'scripts.run_e32', '--source', source]+
            (['--control'] if control else []), stdout=stream, stderr=subprocess.STDOUT, check=True)


def calibration(rows: list[dict]) -> dict:
    """最尤当たり率・多クラスlog loss・最尤確信度別の較正を集計する。"""
    bins = []
    for index in range(CALIBRATION_BINS):
        selected = [r for r in rows if min(int(r['confidence']*CALIBRATION_BINS), CALIBRATION_BINS-1) == index]
        bins.append(dict(lower=index/CALIBRATION_BINS, upper=(index+1)/CALIBRATION_BINS, n=len(selected),
            confidence=sum(r['confidence'] for r in selected)/len(selected) if selected else None,
            accuracy=sum(r['hit'] for r in selected)/len(selected) if selected else None))
    return dict(n=len(rows), hits=sum(r['hit'] for r in rows),
        accuracy=sum(r['hit'] for r in rows)/len(rows) if rows else None,
        log_loss=sum(r['log_loss'] for r in rows)/len(rows) if rows else None, bins=bins)


def report() -> dict:
    """事前登録の母数・閾値を変えず、較正とE31比較を追加する。"""
    e31.OUT = OUT
    value = e31.report()
    paths = {s:e31.prior.directory('on',s)/'snapshot_final_audit.json' for s in SOURCES}
    audits = {s:json.loads(p.read_text()) for s,p in paths.items()}
    rows = [r for v in audits.values() for r in v['calibration']]
    save_json(OUT/'CALIBRATION.json', dict(three_videos=calibration(rows),
        sources={s:calibration(v['calibration']) for s,v in audits.items()}))
    save_json(OUT/'ENUMERATION_TIMING.json', e31.quantiles([
        r['enumeration_sec'] for v in audits.values() for r in v['rows']]))
    save_json(OUT/'COMPARISON.json', dict(e27=json.loads(Path('logs/e27/on/METRICS.json').read_text()),
        e31=json.loads(Path('logs/e31/on/METRICS.json').read_text()), e32=value))
    return value


def main() -> None:
    """ON/OFFを全記録で実行してから集約する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=e31.prior.ALL_SOURCES)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
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
