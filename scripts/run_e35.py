"""E35の5固定記録を本番構成と同じ採点行で再生する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from scripts import run_e32 as baseline
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_overlay import ExchangeEventOverlay

OUT = Path('logs/e35')
SOURCES = ('review', *baseline.e31.prior.ALL_SOURCES[:-1])


def worker(source: str, control: bool = False) -> None:
    """R1系を入力に使わず、E32と同じE31記録の全行を再生する。"""
    prior = baseline.e31.prior
    prior.OUT, prior.AuditTrace = OUT, baseline.BeliefTrace
    timings: list[float] = []
    original = ExchangeEventOverlay.update
    def timed(overlay: Any, *args: Any, **kwargs: Any) -> None:
        started = perf_counter()
        original(overlay, *args, **kwargs)
        timings.append(perf_counter()-started)
    ExchangeEventOverlay.update = timed
    def replay(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        value = replay_module.replay(Path('logs/e31/records')/record.name, dest, *args, **kwargs)
        save_json(dest/'snapshot_final_audit.json', args[2].summary())
        save_json(dest/'TIMING.json', dict(update_sec=baseline.e31.quantiles(timings), total_sec=sum(timings)))
        return value
    prior.replay = replay
    variant = 'off' if control else 'on'
    prior.locked_worker(variant, source, dict(baseline.OPTIONS, post_counter_death_bound=not control))
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', replay_module.compare(Path('logs/e32/on')/suffix,
                                                                prior.directory(variant, source)))


def main() -> None:
    """R1bと同居するため一度に一記録だけ実行し、完成済み結果から再開する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source, args.control)
        return
    for source in SOURCES:
        with (OUT/f'{source}.log').open('a') as log:
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_e35', '--source', source],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    from scripts.report_e35 import report
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
