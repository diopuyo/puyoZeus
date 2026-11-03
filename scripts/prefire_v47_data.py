"""v4.6の因果的候補原票を再利用し、同じ入力時刻で台帳と固定B3を再計算する。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scripts.exchange_review_io import jsonl, save, sha256
from scripts.prefire_v46_data import engine_for as old_engine, OUT as PREVIOUS, VIDEOS, ASSETS, MODEL_HASH
from scripts.prefire_v44_data import notifications, source
from src.prefire_post_v47 import PostV47
from src.prefire_post_b3 import forecast_features, validate_publication, MS, LIMIT_MS
from src.prefire_worker import LOW_SPEC_RATIO

OUT = PREVIOUS.parent/'v4_7'
EVIDENCE = Path('docs/verification/prefire_v47')
PROGRESS = 500


def engine_for() -> PostV47:
    old = old_engine()
    engine = PostV47(old.predictor, old.table, old.policy, old.native, True, old.pre_model)
    old.pool.shutdown()
    return engine


def ledger_row(engine: PostV47, stamp: float, game: int) -> dict:
    snap = engine.state.sl_snap
    values = {name:[snap[f'{name}_p{i+1}'] for i in (0, 1)]
        for name in ('pending', 'generated', 'dropped', 'carry', 'score', 'pending_confirmed')}
    values = {k:[v if v is not None and np.isfinite(v) else None for v in items]
              for k, items in values.items()}
    batches = [[vars(b).copy() for b in v] for v in engine.state.sl.core.batches]
    return dict(t=stamp, game=game, **values, batches=batches, rate=snap['rate'],
        origin=snap['margin_origin_sec'],
        chain_ids=[c['signature'][0] if c else None for c in engine.chains],
        events=[dict(kind=e.kind, side=e.side, t=e.t_event, available=e.available_at_sec,
                     detail=dict(e.detail)) for e in snap['events']])


def recalculate(engine: PostV47, old: dict, projection: dict) -> dict:
    began = perf_counter()
    value = dict(old, **projection)
    # 候補盤・物理探索は既存の因果的原票を固定再利用する。
    value['boards'] = old['boards']
    row = forecast_features(engine.state, value['boards'], value['pending'], engine.table, engine.native)
    value['v_post'] = engine.predictor.predict(row)
    inference = (perf_counter()-began)*MS
    value['pre_records'] = deepcopy(old.get('pre_records'))
    for record in value['pre_records'] or []:
        record['request']['pending'] = int(value['pending'][record['side']])
    reused = max(0., old['low_spec_ms']-old['inference_ms']*LOW_SPEC_RATIO)
    elapsed = reused+(perf_counter()-began)*MS*LOW_SPEC_RATIO+value.pop('projection_cost_ms')
    requested = old.get('requested_sec', old['input_sec'])
    value.update(inference_ms=inference, reused_physics_ms=reused, low_spec_ms=elapsed,
        ready_sec=old['input_sec']+elapsed/MS,
        request_latency_ms=(old['input_sec']-requested)*MS+elapsed,
        over_budget=(old['input_sec']-requested)*MS+elapsed>LIMIT_MS,
        source_record_input=old['input_sec'])
    validate_publication(value)
    return value


def dispatch(engine: PostV47, old: dict) -> dict:
    if 'v_post' not in old:
        if 'closed' in old:
            engine.chains, engine.fight = [None, None], None
        return old
    began = perf_counter()
    for i, active in enumerate(old['board_prediction_active']):
        if engine.chains[i] is not None:
            engine.chains[i]['finished'] = not active
    projection = engine.projection()
    projection['projection_cost_ms'] = (perf_counter()-began)*MS*LOW_SPEC_RATIO
    return recalculate(engine, old, projection)


def collect(video: int, limit: int | None = None) -> None:
    records = {r['input_sec']:r for r in jsonl(PREVIOUS/f'predictions_{video}.jsonl')}
    games = {r['game'] for r in records.values()}
    origins = json.loads((PREVIOUS/f'origins_{video}.json').read_text())
    engine, cursor, count, journal, signature = engine_for(), 0, 0, [], None
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/f'predictions_{video}.jsonl').open('w') as stream:
        for stamp, game, raw in notifications(video):
            if game not in games:
                continue
            engine._boundary(stamp, game)
            engine._current(stamp, game, raw)
            delivered = []
            while cursor < len(origins) and origins[cursor]['input_sec'] <= stamp:
                if origins[cursor]['game'] == game:
                    delivered.append(origins[cursor])
                cursor += 1
            engine._accept_forecasts(delivered, stamp, game)
            engine._chains(stamp, raw)
            if engine.state.sl_snap is None:
                continue
            item = ledger_row(engine, stamp, game)
            key = (game, item['pending'], item['generated'], item['dropped'], item['carry'], item['origin'])
            if key != signature or item['events'] or stamp in records:
                journal.append(item)
                signature = key
            if stamp not in records:
                continue
            value = dispatch(engine, records[stamp])
            stream.write(json.dumps(value)+'\n')
            count += 1
            if count % PROGRESS == 0:
                stream.flush()
                print('v47', video, count, flush=True)
            if limit and count >= limit:
                break
    engine.pool.shutdown()
    save(OUT/f'ledger_{video}.json', journal)
    save(OUT/f'predictions_{video}.receipt.json', dict(rows=count, input_sha256=sha256(source(video)/'inputs.jsonl.gz'),
        output_sha256=sha256(OUT/f'predictions_{video}.jsonl'), source_sha256=sha256(PREVIOUS/f'predictions_{video}.jsonl'),
        native_sha256=sha256(Path(engine.native.__file__)), model_sha256=MODEL_HASH))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=int, nargs='+', default=list(VIDEOS))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if set(args.videos)&{80, 52}:
        assert (EVIDENCE/'fit_lock.json').exists(), 'レビュー前にfitを固定'
    for video in args.videos:
        collect(video, args.limit)


if __name__ == '__main__':
    main()
