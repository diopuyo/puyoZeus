"""製品コードを変更せず、同一フラグ再生でS3実入力を観測する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import shlex
import time
from typing import Any
import numpy as np

from scripts._diag_review1003b_extract import ROOT, OUT, CASES, resources
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_evaluator import ExchangeEndInput, build_features, count_sides
from src.exchange_event_count_features import orient
from src.exchange_event_record import encode

ALIASES = {'exchange_event_live_count': 'live_count', 'exchange_event_model_dir': 'model_dir',
           'exchange_event_death_guard': 'death_guard', 'exchange_event_switch_smoothing': 'switch_smoothing'}
RESOURCE_INTERVAL = 20.


def options() -> dict[str, Any]:
    """納品manifestの引数をそのまま再生へ渡す。"""
    tokens = shlex.split(json.loads((ROOT / 'manifest.json').read_text())['production_flags'])
    result = {}
    while tokens:
        name = tokens.pop(0).removeprefix('--').replace('-', '_')
        value = tokens.pop(0) if tokens and not tokens[0].startswith('--') else True
        name = ALIASES.get(name, name)
        if name == 'exchange_event_update':
            continue
        assert name in inspect.signature(replay).parameters, name
        if name.endswith('_dir'):
            value = Path(value)
        elif name == 'hidden_scenario_cap':
            value = int(value)
        result[name] = value
    return result


def observation(overlay: Any, args: tuple) -> dict[str, Any]:
    """凍結特徴・live特徴・完走予測・盤面履歴を複製する。"""
    tracker = overlay.tracker
    row = dict(t_sec=args[3], source=tracker.source, p1=tracker.probability,
        history=[dict(t_sec=h[-1].t_sec, board=h[-1].board, queue=h[-1].queue)
                 if h else None for h in overlay._history])
    if tracker.current is not None:
        row['chains'] = [asdict(c) for c in tracker.current.chains]
    if tracker.firing is not None:
        totals = np.array([sum(c.provisional_score for c in tracker.current.chains
            if c.side == side) for side in ('1P', '2P')])
        event = ExchangeEndInput(tracker.firing, np.zeros(2), totals, tracker._score_elapsed)
        _, features, side = build_features(event, tracker.models.elapsed_thresholds)
        row.update(firing=asdict(tracker.firing), totals=totals,
            features=np.r_[features, orient(count_sides(event), side)],
            score_elapsed=tracker._score_elapsed)
        from src.exchange_hidden_row_probability import scenarios
        if tracker.hidden_row_belief is not None:
            row['score_scenarios'] = scenarios(tracker.hidden_row_belief, tracker.current.chains)
    return encode(row)


def main() -> None:
    """資源条件を定期確認して3試合を逐次再生する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=int, choices=(4, 6, 7))
    case = next(c for c in CASES if c[0] == parser.parse_args().case)
    number, set_no, game, start, end = case
    dest = OUT / f'replay_case{number}'
    dest.mkdir(exist_ok=True)
    resources()
    last_check = [0.]
    with (dest / 'observations.jsonl').open('w') as stream:
        def observer(overlay: Any, args: tuple) -> None:
            if time.monotonic()-last_check[0] >= RESOURCE_INTERVAL:
                print(json.dumps(dict(t_sec=args[3], **resources())), flush=True)
                last_check[0] = time.monotonic()
            if start <= args[3] <= end:
                stream.write(json.dumps(observation(overlay, args))+'\n')
                stream.flush()
        replay(ROOT / 'work' / f'set{set_no}_game{game:02}' / 'inputs.jsonl.gz', dest,
               observer=observer, **options())


if __name__ == '__main__':
    main()
