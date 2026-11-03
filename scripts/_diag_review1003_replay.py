"""本番入力を同一フラグで再生し、評価入力を観測する（製品コード無変更）。"""
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
from scripts._diag_review1003_extract import ROOT, OUT, WINDOWS, resources
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_evaluator import ExchangeEndInput, build_features, count_sides
from src.exchange_event_record import encode

ALIASES = {'exchange_event_live_count': 'live_count',
           'exchange_event_model_dir': 'model_dir',
           'exchange_event_death_guard': 'death_guard',
           'exchange_event_switch_smoothing': 'switch_smoothing'}
RESOURCE_INTERVAL = 20.0


def options() -> dict[str, Any]:
    """納品時manifestから同一の本番引数を復元する。"""
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


def observation(overlay: Any, args: tuple) -> dict:
    """モデルへ供給された特徴と盤面を複製する。"""
    tracker = overlay.tracker
    row = dict(t_sec=args[3], source=tracker.source, p1=tracker.probability,
               history=[dict(t_sec=h[-1].t_sec, board=h[-1].board, queue=h[-1].queue)
                        if h else None for h in overlay._history])
    if tracker.current is not None:
        row['chains'] = [asdict(c) for c in tracker.current.chains]
        row['exchange_id'] = tracker.current.exchange_id
    if tracker.firing is not None:
        totals = np.array([sum(c.provisional_score for c in tracker.current.chains
                              if c.side == side) for side in ('1P', '2P')])
        event = ExchangeEndInput(tracker.firing, np.zeros(2), totals, tracker._score_elapsed)
        row['firing'] = asdict(tracker.firing)
        row['score_totals'] = totals
        row['score_elapsed'] = tracker._score_elapsed
        row['features'] = build_features(event, tracker.models.elapsed_thresholds)[1]
        row['count_sides'] = count_sides(event)
    return encode(row)


def main() -> None:
    """逐次再生し、空きメモリ条件と時系列入力を記録する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=int, choices=[w[0] for w in WINDOWS])
    game = parser.parse_args().game
    _, start, end = next(w for w in WINDOWS if w[0] == game)
    dest = OUT / f'replay_game{game:02}'
    dest.mkdir(exist_ok=True)
    last_check = [0.0]
    with (dest / 'observations.jsonl').open('w') as stream:
        def observer(overlay: Any, args: tuple) -> None:
            if time.monotonic() - last_check[0] >= RESOURCE_INTERVAL:
                with (dest / 'resources.jsonl').open('a') as log:
                    log.write(json.dumps(dict(t_sec=args[3], **resources())) + '\n')
                last_check[0] = time.monotonic()
                print(game, args[3], flush=True)
            if start <= args[3] <= end:
                stream.write(json.dumps(observation(overlay, args)) + '\n')
                stream.flush()
        result = replay(ROOT / 'work' / f'set1_game{game:02}' / 'inputs.jsonl.gz',
                        dest, observer=observer, **options())
    (dest / 'result.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main()
