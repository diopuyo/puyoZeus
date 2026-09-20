"""実保存された消去前familyの診断投影。別runや終端盤面を補完に使わない。"""
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
OBSERVED_FRAME, FRAME, TOKEN = 35662, 35672, 'step:6620'


def rows(name: str) -> Any:
    with (SOURCE / name).open() as stream:
        for line in stream:
            yield json.loads(line)


def inputs() -> tuple[Any, Any, dict, dict]:
    observation = next(r for r in rows('PREFIX_LANE_OBSERVATIONS.jsonl') if r['frame'] == OBSERVED_FRAME)
    assert len(observation['families']) == 1
    family = observation['families'][0]
    assert family['phase'] == 'prepop' and family['synthetic_derived'] is False
    value = S.decode(family['state'])
    assignment = next(r for r in rows('PREFIX_ORIGIN_ASSIGNMENTS.jsonl')
        if r.get('candidate') and r['candidate']['source_frame'] == FRAME)
    candidate = assignment['candidate']
    assert candidate['observed_frame'] == value.frame == OBSERVED_FRAME
    assert candidate['source_call_token'] == TOKEN and candidate['non_binding'] is True
    journal = next(r for r in rows('atomic_journal.jsonl') if r.get('token') == TOKEN)
    assert journal['frame_idx'] == FRAME and journal['side'] == '1P' and journal['exception'] is None
    assert journal['source_id'] == value.scope[0] and journal['run_id'] == value.scope[1]
    origins = [e['active_origin'] for e in journal['events'] if e.get('active_origin') is not None
        and e['active_origin']['object_id'] == candidate['origin_object_id']]
    assert origins and all(o['trigger_sec'] == candidate['origin_trigger_sec'] for o in origins)
    assert all(o['before_board']['grid'] == origins[0]['before_board']['grid'] for o in origins)
    proof = dict(observation=observation, assignment=assignment, source_step=journal)
    return value, P.B.Board.from_dict(origins[0]['before_board']), candidate, proof


def main() -> None:
    started = time.perf_counter()
    value, origin, candidate, proof = inputs()
    projected = P.project(value, value.scope, origin, origin_frame=FRAME, cutoff_frame=FRAME,
        origin_token=TOKEN, origin_trigger_sec=candidate['origin_trigger_sec'])
    receipt = dict(source_run=str(SOURCE), source_frame=value.frame, cutoff_frame=FRAME,
        source_worlds=len(value.worlds), outcomes=len(projected.outcomes),
        chain_counts=sorted({o.chain_count for o in projected.outcomes}),
        raw_chain_scores=sorted({o.raw_chain_score for o in projected.outcomes}),
        probability_mass=sum(o.weight for o in projected.outcomes),
        evidence_sha256=hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest(),
        projected_code_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        absolute_arrival_index=candidate['absolute_arrival_index'],
        candidate_replayed=False, live_registry_authorized=False, future_observation_used=False,
        physical_GT_verified=False, model_evaluated=False, quality_gate_clear=False,
        seconds=time.perf_counter() - started)
    with (ROOT / 'SAVED_PREPOP_PROJECTION_v1.json').open('x') as stream:
        json.dump(dict(receipt=receipt, projected_view=asdict(projected), source_evidence=proof), stream)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
