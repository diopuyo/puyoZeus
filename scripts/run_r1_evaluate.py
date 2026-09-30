"""固定OFF・再収集ONを本番撃ち合いCLIで評価し、旧E32との全行一致を検収する。"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from scripts import replay_exchange_event_20260926 as replay_module
from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e31 import SnapshotTrace
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from src.exchange_event_tracker import ExchangeEventTracker

OUT = Path('logs/r1')
WORKERS = 3


def worker(source: str, mode: str) -> None:
    """本番構成を個別フラグへ手で転記せず、実際のCLI展開を通す。"""
    prior.OUT = OUT
    dest = prior.directory(mode, source)
    dest.mkdir(parents=True, exist_ok=True)
    reach = prior.runner.runner.reach
    reach.instrument(ExchangeEventTracker)
    reach.STATS, trace = defaultdict(reach.empty_stats), SnapshotTrace(source)
    reach.FRAMES.clear()
    original = replay_module.replay
    def replay(*args: Any, **kwargs: Any) -> dict:
        result = original(*args, **dict(kwargs, observer=trace))
        save_json(dest/'snapshot_final_audit.json', trace.summary())
        save_json(dest/'count_trace.json', trace.rows)
        return result
    replay_module.replay = replay
    record = (Path('logs/e31/records') if mode == 'off' else OUT/'records')/f'{source}.jsonl.gz'
    sys.argv = ['r1_evaluate', str(record), '--out', str(dest), '--production-exchange-event']
    save_json(dest/'COMMAND.json', sys.argv)
    replay_module.main()
    if source in SOURCES:
        rows = reach.event_rows(dest, source)
        save_json(OUT/mode/'v2'/source/'reach.json', dict(rows=rows,
            summary=dict(source=source, frames=dict(reach.FRAMES), **reach.summary(rows))))
    if mode == 'off':
        relative = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(dest/'E32_EQUIVALENCE.json', replay_module.compare(Path('logs/e32/on')/relative, dest))
    save_json(dest/'DONE.json', dict(source=source, mode=mode, completed=True))


def launch(task: tuple[str, str]) -> None:
    """完了済みの再生を繰返さず、一記録一プロセスで実行する。"""
    source, mode = task
    prior.OUT = OUT
    dest = prior.directory(mode, source)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'DONE.json').exists():
        return
    with (dest/'runner.log').open('a') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'scripts.run_r1_evaluate',
            '--source', source, '--mode', mode], stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """全OFF入力の一致を必須にし、完成済みのONだけを採点する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=prior.ALL_SOURCES)
    parser.add_argument('--mode', choices=('off', 'on'))
    args = parser.parse_args()
    assert (OUT/'OFF_INPUT_CHECK.json').exists()
    if args.source:
        worker(args.source, args.mode)
        return
    assert (OUT/'ON_COLLECT_COMPLETE.json').exists()
    for mode in ('off', 'on'):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, [(s, mode) for s in prior.ALL_SOURCES]))
    (OUT/'EVALUATION_COMPLETE.json').write_text(json.dumps(dict(completed=True)))


if __name__ == '__main__':
    main()
