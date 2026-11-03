"""通知を時刻順に読み、STABLE盤と連鎖終了予測盤を同じ専用経路へ配送する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from scripts.exchange_review_io import jsonl, save, sha256
from scripts.prefire_v4_data import source as old_source, ledger, VIDEOS, RANDOM_SEED
from scripts.prefire_v41_fit import load_rows
from src.prefire_growth import SideInput, SideSnapshots, ROLLOUT_COUNT
from src.prefire_branches import calculate, MS
from src.prefire_worker import LOW_SPEC_RATIO
from src.prefire_chain_board import initial, advance
from src.exchange_tsumo_tape import TsumoTape
from src.b3_timing import STAGE_SEC_DEFAULT, TAIL_SEC_DEFAULT

OUT = Path(_public_dev_text('<work>/exch_review_render/v4_4'))
EVIDENCE = Path('docs/verification/prefire_v44')
PREVIOUS = OUT.parent/'v4_3'
HISTORY = 3
CONSUMPTION_LIMIT = .6
MAX_LOW_SPEC_MS = 500.
PAIR_SIZE = 2


def source(video: int) -> Path:
    return PREVIOUS/'input_52' if video == 52 else old_source(video)


def notifications(video: int) -> object:
    for row in jsonl(source(video)/'inputs.jsonl.gz'):
        if row.get('kind') != 'update':
            continue
        args = row['args']['tuple']
        yield float(args[3]), int(args[4]), [args[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')]


def unpack(raw: dict) -> dict:
    board = raw.get('confirmed_board')
    pair = lambda k: raw.get(k, {}).get('tuple') if isinstance(raw.get(k), dict) else raw.get(k)
    return dict(state=raw['state']['state'], board=board['board']['array'] if board else None,
        next=pair('next_pair'), dnext=pair('dnext_pair'), slide=bool(raw.get('next_slide_motion')))


def chain_update(raw: dict, state: dict | None, history: list[dict], known: tuple,
                 colors: tuple, stamp: float, policy: dict, native: object) -> tuple:
    event = raw.get('chain_event')
    if not event or len(colors) != 4:
        return state, None
    event = event['namespace']
    signature = (event['trigger_sec'], event['chain_count'], event['total_score'])
    if state and state['signature'] == signature:
        return state, None
    began = perf_counter()
    if state is None or state['signature'][0] != signature[0]:
        board = np.asarray(event['before_board']['board']['array']).reshape(-1).tolist()
        state = initial(board, history, known, list(colors), stamp, policy, native)
    state['signature'] = signature
    result = advance(state, int(signature[1]), int(signature[2]), stamp, list(colors), policy, native)
    result['chain_ms'] = (perf_counter()-began)*MS*LOW_SPEC_RATIO
    result['busy_until'] = stamp+result['remaining_stages']*STAGE_SEC_DEFAULT+TAIL_SEC_DEFAULT
    return state, result


def context() -> dict:
    return dict(tape=TsumoTape(), snapshots=SideSnapshots(), colors=set(), histories=[[], []],
        chains=[None, None], chain_labels=[None, None], start=0., game=None, pending=None, cursor=0)


def update_context(ctx: dict, stamp: float, game: int, sides: list[dict], pending: list[dict]) -> list[int]:
    if ctx['game'] != game:
        cursor = ctx['cursor']
        ctx.update(context(), cursor=cursor, start=stamp, game=game)
    for side, raw in enumerate(sides):
        ctx['tape'].observe(side, stamp, raw['next'], raw['dnext'], raw['slide'])
        ctx['colors'].update(c for p in ctx['tape'].known_future(side) for c in p)
    while ctx['cursor'] < len(pending) and pending[ctx['cursor']]['t_sec'] <= stamp:
        ctx['pending'] = pending[ctx['cursor']]
        ctx['cursor'] += 1
    return ctx['snapshots'].observe(game, sides)


def label_finished(ctx: dict, side: int, stamp: float, board: list[int], labels: list[dict]) -> None:
    previous = ctx['chain_labels'][side]
    if previous:
        material = sum(1 <= c <= 5 for c in board)
        added = PAIR_SIZE*max(0, ctx['tape'].placements(side)-previous['placements'])
        consumed = previous['material']+added-material
        ratio = consumed/max(1, previous['material'])
        labels.append(dict(previous, end=stamp, consumed=consumed, ratio=ratio,
            provocation=ratio < CONSUMPTION_LIMIT, valid=0 < consumed <= previous['material']))
        ctx['chain_labels'][side] = None
    ctx['chains'][side] = None


def make_request(ctx: dict, side: int, stamp: float, board: list[int], video: int,
                 extra: dict) -> dict:
    pending = ctx['pending']
    request = SideInput(ctx['game'], side, stamp, tuple(board), ctx['tape'].known_future(side)[:3],
        tuple(sorted(ctx['colors'])), int(pending['pending'][side]), stamp-ctx['start'],
        tuple([stamp, pending['t_sec']]+extra.get('evidence_times', [])),
        ctx['tape'].placements(side), RANDOM_SEED+video*1000+ctx['game'])
    return dict(request=asdict(request), video=video, kind=extra.get('kind', 'stable'),
        chain_ms=extra.get('chain_ms', 0.), busy_until=extra.get('busy_until', stamp),
        conflict=extra.get('conflict', False))


def side_requests(ctx: dict, raw: list[dict], sides: list[dict], changed: list[int],
                  stamp: float, video: int, policy: dict, native: object, labels: list[dict]) -> list[dict]:
    result = []
    for side in (0, 1):
        if side in changed:
            board = list(map(int, ctx['snapshots'].boards[side]))
            label_finished(ctx, side, stamp, board, labels)
            ctx['histories'][side] = (ctx['histories'][side]+[dict(t=stamp, board=board)])[-HISTORY:]
            result.append(make_request(ctx, side, stamp, board, video, {}))
        elif sides[side]['state'] == 'CHAIN':
            state, value = chain_update(raw[side], ctx['chains'][side], ctx['histories'][side],
                ctx['tape'].known_future(side), tuple(sorted(ctx['colors'])), stamp, policy, native)
            ctx['chains'][side] = state
            if value:
                if ctx['chain_labels'][side] is None:
                    origin = np.asarray(raw[side]['chain_event']['namespace']['before_board']['board']['array']).reshape(-1).tolist()
                    ctx['chain_labels'][side] = dict(video=video, game=ctx['game'], side=side,
                        trigger=state['signature'][0], placements=ctx['tape'].placements(side),
                        material=sum(1 <= c <= 5 for c in origin))
                result.append(make_request(ctx, side, stamp, value['board'], video, dict(value, kind='chain_end')))
    return result


def collect(video: int, games: set[int]) -> None:
    import puyo_core
    ctx, labels, pending = context(), [], ledger(source(video))
    policy = json.loads(Path('docs/verification/exch_review_v310/policy_locked.json').read_text())
    path = OUT/f'requests_{video}.jsonl'
    count = 0
    with path.open('w') as stream:
        for stamp, game, raw in notifications(video):
            sides = [unpack(r) for r in raw]
            changed = update_context(ctx, stamp, game, sides, pending)
            if game not in games or ctx['pending'] is None or ctx['pending']['game'] != game:
                continue
            for item in side_requests(ctx, raw, sides, changed, stamp, video, policy, puyo_core, labels):
                stream.write(json.dumps(item)+'\n')
                count += 1
            if count and count % 1000 == 0:
                stream.flush()
    save(OUT/f'labels_{video}.json', labels)
    save(OUT/f'requests_{video}.receipt.json', dict(count=count, input_sha256=sha256(source(video)/'inputs.jsonl.gz'),
        requests_sha256=sha256(path), labels=len(labels)))
    print('入力採取', video, count, '連鎖教師', len(labels), flush=True)


def compute(video: int) -> None:
    import puyo_core
    path, busy, game, counts = OUT/f'curves_{video}.jsonl', 0., None, {}
    with path.open('w') as stream, ThreadPoolExecutor(ROLLOUT_COUNT) as pool:
        for index, item in enumerate(jsonl(OUT/f'requests_{video}.jsonl')):
            request = SideInput(**item['request'])
            try:
                value = calculate(request, puyo_core, pool, True)
                value.update({k: item[k] for k in ('kind', 'busy_until', 'conflict', 'chain_ms')})
                value['low_spec_ms'] += item['chain_ms']
                if game != request.game:
                    busy, game = request.input_sec, request.game
                value['ready_sec'] = max(busy, request.input_sec)+value['low_spec_ms']/MS
                busy = value['ready_sec']
                value['over_budget'] = value['low_spec_ms'] > MAX_LOW_SPEC_MS
                assert not value['over_budget'], value['low_spec_ms']
            except ValueError as error:
                value = dict(game=request.game, side=request.side, input_sec=request.input_sec, reason=str(error))
            value['video'] = video
            stream.write(json.dumps(value)+'\n')
            counts[value.get('kind', 'missing')] = counts.get(value.get('kind', 'missing'), 0)+1
            if (index+1) % 500 == 0:
                stream.flush()
                print('物理', video, index+1, counts, flush=True)
    save(path.with_suffix('.receipt.json'), dict(counts=counts, output_sha256=sha256(path),
        native_sha256=sha256(Path(puyo_core.__file__))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=int, nargs='+', default=list(VIDEOS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for video in args.videos:
        if video in (80, 52) and not (EVIDENCE/'model.json').exists():
            raise ValueError('fit固定前のレビュー動画処理は禁止')
        games = {r['game'] for r in rows if r['video'] == video}
        games = {7} if video == 80 else {31, 32} if video == 52 else games
        if not (OUT/f'requests_{video}.receipt.json').exists():
            collect(video, games)
        if not (OUT/f'curves_{video}.receipt.json').exists():
            compute(video)


if __name__ == '__main__':
    main()
