"""保存済み同runの初回起点までだけを使い予測分布を再生。実live/GT検収ではない。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import projected_view as P
import serialization as S

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'video38_split_tail_candidate_v22'
TOKEN, FRAME, SIDE = 'step:6132', 35184, '1P'


def inputs() -> tuple[Any, Any, Any]:
    with (SOURCE / 'PROBABILISTIC_BASIS.jsonl').open() as stream:
        basis = json.loads(next(stream))
    value = S.decode(basis['state'])
    matches = []
    with (SOURCE / 'atomic_journal.jsonl').open() as stream:
        for line in stream:
            if TOKEN in line:
                row = json.loads(line)
                if row.get('token') == TOKEN:
                    matches.append(row)
    assert len(matches) == 1
    row = matches[0]
    assert row['frame_idx'] == FRAME and row['side'] == SIDE and row['exception'] is None
    assert (row['source_id'], row['run_id'], row['software_reset'], row['pipe_object_id'],
        row['generation_after']['reset_epoch'], row['side']) == (
        value.scope[0], value.scope[1], value.scope[2], value.scope[3], value.scope[5], value.scope[6])
    origins = [event['active_origin'] for event in row['events'] if event.get('active_origin') is not None]
    assert origins and len({(o['object_id'], o['trigger_sec']) for o in origins}) == 1
    origin = origins[0]
    assert all(o['before_board']['grid'] == origin['before_board']['grid'] for o in origins)
    assert value.frame / 60 <= origin['trigger_sec'] <= FRAME / 60
    return value, P.B.Board.from_dict(origin['before_board']), row


def main() -> None:
    started = time.perf_counter()
    value, origin, journal = inputs()
    result = P.project(value, value.scope, origin, origin_frame=FRAME, cutoff_frame=FRAME, origin_token=TOKEN)
    targets = ('PROBABILISTIC_BASIS.jsonl', 'atomic_journal.jsonl')
    receipt = dict(source_run=str(SOURCE), source_files={name:
        hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in targets},
        basis_frame=value.frame, source_worlds=len(value.worlds), journal_token=TOKEN, cutoff_frame=FRAME,
        outcomes=len(result.outcomes), probability_mass=sum(o.weight for o in result.outcomes),
        chain_counts=sorted({o.chain_count for o in result.outcomes}),
        raw_chain_scores=sorted({o.raw_chain_score for o in result.outcomes}),
        source_J_scope_matched=True, future_observation_used=False, live_registry_authorized=False,
        physical_GT_verified=False, model_evaluated=False, quality_gate_clear=False,
        seconds=time.perf_counter() - started)
    with (ROOT / 'SAVED_ORIGIN_PROJECTION_v1.json').open('x') as stream:
        json.dump(dict(receipt=receipt, projected_view=asdict(result)), stream)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
