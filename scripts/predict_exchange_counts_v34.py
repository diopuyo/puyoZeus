"""固定した設定で初回予測。正解・神託・発火後の盤面を開かない。"""
from __future__ import annotations

from copy import deepcopy
import json
from time import perf_counter
from scripts.audit_exchange_counts_v33 import OUT as V33
from scripts.audit_exchange_cells_v34 import OUT
from scripts.exchange_review_v32_data import BASE, VIDEOS
from scripts.exchange_review_io import save, sha256
from scripts.predict_exchange_counts_v33 import with_send
from src.exchange_count_prediction import distribution, LOW_SPEC_RATIO
from src.exchange_count_v34 import candidates, condition, MAX_LOW_SPEC_MS

MS = 1000


def predict(inputs: dict, policy: dict, native: object) -> dict:
    began = perf_counter()
    stamp, sources = inputs['input_sec'], inputs['sources']
    out = candidates(sources, inputs['legacy'], inputs['colors'], policy, stamp,
                     native, inputs['history'], inputs['pair'])
    accepted, conflict = condition(out['candidates'], [inputs['first']], stamp)
    ledger = inputs['ledger']
    if ledger['observed_sec'] > stamp:
        raise ValueError('未来の得点台帳')
    values = distribution(with_send(deepcopy(accepted or out['candidates']), stamp, ledger), policy['temperature'])
    elapsed = (perf_counter()-began)*MS+sum(s.get('image_compute_ms', 0.) for s in sources)
    low_spec = elapsed*LOW_SPEC_RATIO
    if low_spec > MAX_LOW_SPEC_MS:
        raise RuntimeError(f'予測時間超過: {low_spec:.2f}ms')
    return dict(event_id=inputs['event_id'], input_sec=stamp, first=inputs['first'], ledger=ledger,
        variants=dict(board_candidates=values), conflicts=dict(board_candidates=conflict),
        elapsed_ms=elapsed, low_spec_ms=low_spec, candidates=out['candidates'],
        evidence_times=out['evidence_times']+[inputs['first']['t'], ledger['observed_sec']],
        status='分布', exhausted=False)


def run() -> None:
    import puyo_core
    policy = json.loads((OUT/'policy.json').read_text())
    assert policy['training_videos'] == list(VIDEOS) and policy['excluded_video'] == 80 and policy['locked']
    sources = {r['event_id']: r['sources'] for r in json.loads((OUT/'inputs_80.json').read_text())}
    histories = {r['event']: r for r in json.loads((OUT/'history_80.json').read_text())}
    prefix = {r['event_id']: r for r in json.loads((V33/'predictions.json').read_text())['rows']}
    items = json.loads((BASE/'video_80_g7_v3_2/events.json').read_text())['origins']
    rows, inputs = [], []
    for item in items:
        eid = item['event_id']
        previous = prefix[eid]
        if not previous['variants']:
            rows.append(dict(event_id=eid, input_sec=previous['input_sec'], variants={}, status=previous['status']))
            continue
        inp = dict(event_id=eid, input_sec=previous['input_sec'], sources=sources[eid],
            history=histories[eid], pair=item['origin'].get('pair_metadata', {}),
            legacy=item['candidates'], colors=item['fire']['exchange_input']['colors'],
            first=previous['first'], ledger=previous['ledger'])
        inputs.append(inp)
        rows.append(predict(inp, policy, puyo_core))
        print('予測', eid, rows[-1]['low_spec_ms'], flush=True)
    save(OUT/'prediction_inputs.json', inputs)
    save(OUT/'predictions.json', dict(rows=rows, policy_sha256=sha256(OUT/'policy.json')))


if __name__ == '__main__':
    run()
