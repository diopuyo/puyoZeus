"""D5固定5記録と、独立したR1 ON先行確認を単一ワーカーで再生する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_e32 as baseline
from scripts import replay_exchange_event_20260926 as replay_module
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.exchange_event_record import encode

OUT = Path('logs/d5')
SOURCES = ('q_7gc4TgFig', 'review', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')
SCENE_START, SCENE_END = 2750., 2775.


class Trace(baseline.BeliefTrace):
    """R1の予測採用と確定時刻を、状態変化ごとに記録する。"""
    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.timeline: list[dict] = []
        self.previous: str | None = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        if not SCENE_START <= inputs[3] <= SCENE_END or self.source not in ('review', 'zenchi'):
            return
        projection, tracker = overlay._landing_projection, overlay.tracker
        record = tracker.current or projection.death_record
        chains = [dict(side=c.side, chain=c.chain_id, trigger=c.trigger_sec,
            score=c.predicted_final_score, count=c.predicted_chain_count, formula=c.formula_total,
            end=c.end_signal_sec, board=c.predicted_final_board) for c in record.chains] if record else []
        value = dict(source=tracker.source, p1=tracker.probability, chains=chains,
            last=projection.last, ledger=projection.safety.ledger.pending,
            counts=projection.counts, states=[inputs[0].p1.state.name, inputs[0].p2.state.name])
        key = json.dumps(encode(value), sort_keys=True)
        if key != self.previous:
            self.timeline.append(dict(t_sec=inputs[3], **value))
            self.previous = key


def worker(source: str, r1: bool = False, control: bool = False) -> None:
    """R1 ONと固定E31入力の出力を分離する。"""
    from scripts.d5_runtime import install
    install()
    prior = baseline.e31.prior
    prior.OUT, prior.AuditTrace = OUT, Trace
    records = Path('logs/r1/records' if r1 else 'logs/e31/records')
    variant = 'r1_e35' if r1 else ('off' if control else 'on')
    def enriched(record: Path, dest: Path, *args: Any, **kwargs: Any) -> dict:
        result = replay_module.replay(records/record.name, dest, *args, **kwargs)
        trace = args[2]
        save_json(dest/'snapshot_final_audit.json', trace.summary())
        save_json(dest/'scene_timeline.json', encode(trace.timeline))
        return result
    prior.replay = enriched
    options = dict(baseline.OPTIONS, post_counter_death_bound=True)
    if not r1 and not control:
        options['single_death_proof_guard'] = True
    prior.locked_worker(variant, source, options)
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', replay_module.compare(
            Path('logs/e35/on')/suffix, prior.directory(variant, source)))


def main() -> None:
    """既定OFFを保ち、検収用ONとR1先行確認を順次起動する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES)
    parser.add_argument('--r1', action='store_true')
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    if args.source:
        worker(args.source, args.r1, args.control)
        return
    tasks = [('review', ['--r1']), ('q_7gc4TgFig', [])]
    tasks += [(s, []) for s in SOURCES if s != 'q_7gc4TgFig']
    tasks += [('review', ['--control'])]
    for source, flags in tasks:
        with (OUT/f'{source}{"_".join(flags)}.log').open('a') as stream:
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_d5', '--source', source, *flags],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
