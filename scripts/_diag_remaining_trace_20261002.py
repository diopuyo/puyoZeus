"""評価の返り値を変えず、対象時刻の内部入力だけを別出力へ保存する。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from scripts._diag_remaining_20261002 import OUT, MIN_MEMORY
from scripts._diag_set2_falsedeath_replay import plain
from scripts.falsedeath_replay_20261001 import input_record

TARGETS = {'mia8KCjr52g': (474.73333333333335, 772.7666666666667, 847.8),
           'c5': (1728.1,), 'zenchi': (2761.0, 2771.15), 'review': (2761.0, 2771.15)}
TOLERANCE = 0.001
PAD = 0.1


def write(stream: Any, row: dict) -> None:
    """観測をJSONLで逐次保存する。"""
    stream.write(json.dumps(plain(row), ensure_ascii=False, default=str)+'\n')


def install(source: str, stream: Any) -> None:
    """元関数を一回呼び、戻り値には触れずに観測する。"""
    from src import exchange_death_inputs as death
    from src.exchange_event_overlay import ExchangeEventOverlay
    from src.exchange_post_counter_bound import PostCounterDeathBound
    original, update, prove = death.death_inputs, ExchangeEventOverlay.update, PostCounterDeathBound.prove
    current = [0.0]

    def inputs(projection: Any, overlay: Any, latest: tuple, incoming: list) -> dict:
        context = original(projection, overlay, latest, incoming)
        if any(abs(current[0]-t) < TOLERANCE for t in TARGETS[source]):
            write(stream, dict(kind='death_inputs', t_sec=current[0], context=context,
                  latest=latest, chains=projection.safety.chains, ledger=projection.safety.ledger,
                  guards=projection.safety.guards, elapsed=projection.safety.elapsed))
        return context

    def traced_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        current[0] = args[3]
        value = update(self, *args, **kwargs)
        if any(abs(current[0]-t) < TOLERANCE for t in TARGETS[source]):
            write(stream, dict(kind='update', t_sec=current[0], args=args,
                  source=self.tracker.source, probability=self.tracker.probability,
                  chains=[c for r in self.tracker.records if r.game_idx == self._game for c in r.chains],
                  ledger=self._landing_projection.safety.ledger))
        return value

    def traced_prove(self: Any, *args: Any, **kwargs: Any) -> dict:
        value = prove(self, *args, **kwargs)
        if source in ('zenchi', 'review') and 2760 <= current[0] <= 2772:
            write(stream, dict(kind='post_counter_call', t_sec=current[0], idx=args[2],
                  result=value, motion=self.next_motion, hands=args[5]))
        return value

    death.death_inputs, ExchangeEventOverlay.update, PostCounterDeathBound.prove = inputs, traced_update, traced_prove


def main() -> None:
    """単一記録を対象直後まで再生し、全履歴を保持する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=tuple(TARGETS))
    parser.add_argument('mode', choices=('prod', 'all', 'no_d3', 'no_d2'))
    args = parser.parse_args()
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(memory['MemAvailable'].split()[0])*1024 >= MIN_MEMORY
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    destination = OUT / f'trace_{args.source}_{args.mode}'
    destination.mkdir(exist_ok=False)
    from scripts import replay_exchange_event_20260926 as replay
    original = replay.read_records

    def limited(path: Path) -> Any:
        for row in original(path):
            if row['kind'] == 'update' and row['args'][3] > max(TARGETS[args.source])+PAD:
                break
            yield row

    replay.read_records = limited
    enabled = [] if args.mode == 'prod' else ['--death-input-integrity', '--post-counter-input-integrity', '--score-attribution']
    if args.mode == 'no_d3':
        enabled.remove('--score-attribution')
    if args.mode == 'no_d2':
        enabled.remove('--post-counter-input-integrity')
    sys.argv = ['replay', str(input_record(args.mode, args.source)), '--out', str(destination),
                '--production-exchange-event', *enabled]
    with (destination / 'trace.jsonl').open('x', encoding='utf-8') as stream:
        install(args.source, stream)
        replay.main()


if __name__ == '__main__':
    main()
