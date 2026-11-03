"""固定済みのv3.3/v3.4を同じ発火前入力へ適用。正解盤は読まない。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from time import perf_counter
from scripts.exchange_v35_manifest import OUT, read, V33, V34
from scripts.exchange_review_io import save, sha256
from scripts.predict_exchange_counts_v34 import predict
from scripts.predict_exchange_counts_v33 import with_send, formula_step
from src.exchange_count_prediction import candidates, condition, distribution, LOW_SPEC_RATIO
from src.exchange_count_ledger import ScorePrefix

FORMULA_TOLERANCE, MAX_MS, MS = .1, 500, 1000


def make_input(video: int, item: dict, ocr: dict, policy: dict) -> dict:
    fire = item['fire']
    side = fire['exchange_input']['attacker']
    first = next((s for s in ocr['steps'] if s['side'] == side and
                  abs(s['t_sec']-fire['trigger_sec']) <= FORMULA_TOLERANCE), None)
    if first is None:
        raise ValueError('通知近傍の初段式が未確認')
    stamp = max(fire['t_sec'], first['t_sec'])
    ledger = ScorePrefix(policy['first_move_quiet_sec'])
    for record in ocr['records']:
        if record['t'] > stamp:
            break
        ledger.observe(record['side'], record['t'], record['score'])
    snapshot = ledger.snapshot(side, stamp)
    if snapshot['carry'] is None or snapshot['first_move'] is None:
        raise ValueError('試合開始の得点0または初手が未確定')
    return dict(video=video, game=fire['game_idx'], event_id=fire['event_id'],
        input_sec=stamp, trigger_sec=fire['trigger_sec'], side=side,
        sources=item['sources'], history=item['history'], legacy=[], pair={},
        colors=fire['exchange_input']['colors'], first=formula_step(first, 1), ledger=snapshot)


def baseline(inputs: dict, policy: dict, native: object) -> dict:
    began = perf_counter()
    out = candidates(inputs['sources'], [], inputs['colors'], policy, inputs['input_sec'], native)
    accepted, conflict = condition(out['candidates'], [inputs['first']], inputs['input_sec'])
    values = distribution(with_send(deepcopy(accepted or out['candidates']), inputs['input_sec'], inputs['ledger']),
                          policy['temperature'])
    elapsed = (perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in inputs['sources'])
    low_spec = elapsed*LOW_SPEC_RATIO
    if low_spec > MAX_MS:
        raise RuntimeError(f'v3.3予測時間超過: {low_spec:.2f}ms')
    return dict(variants=dict(board_candidates=values), low_spec_ms=low_spec,
        conflict=conflict, candidates=out['candidates'],
        evidence_times=out['evidence_times']+[inputs['first']['t'], inputs['ledger']['observed_sec']])


def run(video: int) -> None:
    import puyo_core
    policies = {'v33': read(V33/'policy.json'), 'v34': read(V34/'policy.json')}
    manifest = read(OUT/'manifest.json')
    entry = next(r for r in manifest['selected'] if r['video'] == video)
    for name, path in (('v33', V33), ('v34', V34)):
        assert sha256(path/'policy.json') == manifest['policy_sha256'][name]
    for game in entry['games']:
        folder = OUT/f'video_{video}/game_{game}'
        if (folder/'predictions.json').exists():
            continue
        ocr, output, skipped = read(folder/f'holdout_{video}_ocr.json'), [], []
        for item in read(folder/'inputs.json'):
            try:
                inp = make_input(video, item, ocr, policies['v34'])
            except ValueError as error:
                skipped.append(dict(event_id=item['fire']['event_id'], reason=str(error)))
                continue
            output.append(dict(inputs=inp, v33=baseline(inp, policies['v33'], puyo_core),
                               v34=predict(inp, policies['v34'], puyo_core)))
        save(folder/'predictions.json', dict(rows=output, skipped=skipped,
             policy_sha256=manifest['policy_sha256']))
        print('予測完了', video, game, len(output), skipped, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=int, required=True)
    run(parser.parse_args().video)
