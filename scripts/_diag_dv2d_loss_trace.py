"""履歴を先頭から再生し、同一入力のE35一つ抜きと診断入力を保存する。"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any

from scripts._diag_dv2d_loss_inventory import BASE, OUT, read, resource
from scripts._diag_set2_falsedeath_replay import plain, install_progress
from scripts.falsedeath_replay_20261001 import input_record, enabled_flags, limit_replay

PAD = 0.05
INPUT_INTERVAL = 1.0


def install(rows: list[dict], stream: Any) -> None:
    """元の呼出し結果を返し、代替計算は独立した証明器で行う。"""
    from src.exchange_post_counter_bound import PostCounterDeathBound
    from src import exchange_death_inputs as inputs
    from src.exchange_event_landing import ExchangeLandingProjection
    original, evaluate, death_inputs = PostCounterDeathBound.prove, ExchangeLandingProjection._evaluate, inputs.death_inputs
    state: dict = dict(stamp=None, last={}, context=None, alternatives=[])

    def selected(stamp: float) -> list[dict]:
        return [r for r in rows if r['first_sec']-PAD <= stamp < r['candidate_observed_death_sec']]

    def context(projection: Any, overlay: Any, latest: tuple, incoming: list) -> dict:
        value = death_inputs(projection, overlay, latest, incoming)
        state['context'] = value
        return value

    def prove(self: Any, *args: Any, **kwargs: Any) -> dict:
        value = original(self, *args, **kwargs)
        stamp, idx = args[7], args[2]
        if selected(stamp) and self.current_puyo_integrity_v2c:
            alternate = copy.copy(self)
            alternate.audit, alternate.cache, alternate.board_cache = [], {}, {}
            alternate.current_puyo_integrity_v2c = False
            result = original(alternate, *args, **kwargs)
            state['alternatives'].append(dict(side=f'{idx+1}P', result=result))
        return value

    def traced(self: Any, *args: Any, **kwargs: Any) -> dict:
        stamp = args[6]
        state.update(stamp=stamp, context=None, alternatives=[])
        value = evaluate(self, *args, **kwargs)
        active = selected(stamp)
        if active:
            detail = dict(t_sec=stamp, value=value, no_e35=state['alternatives'])
            key = tuple((r['group'], r['game']) for r in active)
            if stamp-state['last'].get(key, float('-inf')) >= INPUT_INTERVAL:
                overlay = args[0]
                detail.update(context=state['context'], latest=args[2], snapshot=args[1],
                    guards=self.safety.guards, ledger=self.safety.ledger,
                    chains=list(self.safety.chains.values()), elapsed=overlay.tracker._score_elapsed)
                state['last'][key] = stamp
            stream.write(json.dumps(plain(detail), ensure_ascii=False, default=str)+'\n')
        return value

    PostCounterDeathBound.prove, ExchangeLandingProjection._evaluate, inputs.death_inputs = prove, traced, context


def main() -> None:
    """各記録の最終対象まで前置履歴を省かず再生する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--mode', default='trace', choices=('trace', 'no_d3', 'no_d4', 'no_d1', 'no_e35', 'no_h1',
        'no_d1+no_h1', 'no_e35+no_h1', 'no_d3+no_h1', 'no_d4+no_h1',
        'no_d1+no_d3', 'no_d1+no_d3+no_h1'))
    args = parser.parse_args()
    rows = [r for r in read(OUT/'EVENTS.json') if r['record'] == args.source]
    dest = OUT / f'{args.mode}_{args.source}_r2'
    dest.mkdir(exist_ok=False)
    (dest/'RESOURCE.json').write_text(json.dumps(resource()))
    flags = enabled_flags('v2d')
    removed = dict(no_d1='--death-input-integrity', no_d3='--score-attribution-v2',
                   no_e35='--post-counter-current-puyo-integrity-v2c', no_h1='--death-history-integrity')
    for mode in args.mode.split('+'):
        if mode in removed:
            flags.remove(removed[mode])
    record = input_record('prod' if 'no_d4' in args.mode else 'v2d', args.source)
    from scripts import replay_exchange_event_20260926 as replay
    limit_replay(replay, max(r['candidate_observed_death_sec'] for r in rows)+PAD)
    install_progress()
    sys.argv = ['replay', str(record), '--out', str(dest), '--production-exchange-event', *flags]
    (dest/'COMMAND.json').write_text(json.dumps(sys.argv))
    with (dest/'trace.jsonl').open('x', encoding='utf-8') as stream:
        if args.mode == 'trace':
            install(rows, stream)
        replay.main()


if __name__ == '__main__':
    main()
