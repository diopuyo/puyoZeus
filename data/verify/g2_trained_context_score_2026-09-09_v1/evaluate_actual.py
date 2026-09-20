"""同run保存の全時点を結合し、受理時点だけ固定fold6モデルで採点する。"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import score as P
import saved_context as S

INPUT = P.ROOT.parent / 'video38_floating_single_live_2026-09-09_v1'
COMPLETE_SHA = 'edf81349d4f4df737090c72b9885f492ee5e0305b19df1d0df91c4c50bbbbbe2'
LOADER_SHA = '2fd2f8cefc9d0bd513480e884b05164e232b52862b7552a9b22a52dfd8670d0f'
SAMPLE_COUNT = 256
RNG_SEED = 20260909


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def guards(loader: Any) -> dict[str, str]:
    paths = {Path(__file__), P.ROOT / 'SCORER_CONTRACT.md', P.ROOT / 'LOADER_RECEIPT.md'}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, '__file__', None) and Path(m.__file__).is_absolute()
        and Path(m.__file__).suffix == '.py' and Path(m.__file__).resolve().is_relative_to(P.PROJECT))
    return {**loader.guards(), **loader.input_guards(), **{str(p): S.sha(p) for p in paths}}


def analyze(registration: dict[str, Any], members: tuple[Any, ...]) -> dict[str, Any]:
    index = S.indices(INPUT)
    holds, faults, accepted = Counter(), Counter(), []
    count = 0
    for row in S.rows(INPUT / 'provisional_context.jsonl'):
        S.saved_join(row, index)
        count += 1
        try:
            bound = P.B.bind_provisional_context(row, registration)
        except P.B.ContextHold as error:
            holds[str(error)] += 1
            continue
        except P.B.ContextFault as error:
            faults[str(error)] += 1
            continue
        result = P.evaluate(bound, members, sample_count=SAMPLE_COUNT, seed=RNG_SEED)
        accepted.append({'frame': row['frame_idx'], 'result': result,
            'inputs': {key: P.B.digest(getattr(bound.inputs, key).tolist()) for key in
                ('boards', 'queues', 'ledger_values', 'ledger_availability')}})
    P.B.require(count == 3624 and not faults, 'update_count_or_binding_fault')
    return {'updates': count, 'accepted_count': len(accepted), 'accepted': accepted,
        'holds': dict(holds), 'faults': dict(faults), 'same_run_join_all_updates': True,
        'sample_count_requested': SAMPLE_COUNT, 'rng_seed': RNG_SEED,
        'quality_gate_clear': False, 'm1_residual_enabled': False, 'calibration_quality_measured': False}


def main() -> int:
    output = P.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    P.B.require(S.sha(INPUT / 'COMPLETE') == COMPLETE_SHA, 'actual_run_identity')
    P.B.require(S.sha(P.ROOT / 'loader.py') == LOADER_SHA, 'frozen_loader_changed')
    before, receipt = S.verify_complete(INPUT)
    registration = S.registration(INPUT, receipt)
    loader = P.load_loader()
    members = loader.load_members(registration['source_id'])
    before.update(guards(loader))
    report = analyze(registration, members)
    loader.verify_members(registration['source_id'], members)
    after = {path: S.sha(Path(path)) for path in before}
    import torch
    P.B.require(before == after and not torch.cuda.is_initialized(), 'source_or_device_changed')
    write(output / 'REPORT.json', report)
    metadata = [{key: value for key, value in asdict(m).items() if key != 'model'} for m in members]
    write(output / 'MEMBERS.json', metadata)
    result = {'pid': os.getpid(), 'actual_exit': 0, 'seconds': time.perf_counter() - started,
        'before': before, 'after': after, 'actual_run': str(INPUT), 'registration': registration,
        'trained_model_inference': True, 'device': 'cpu', 'quality_gate_clear': False,
        'production_permission': False, 'upstream_publication_unchanged': True}
    write(output / 'RESULT.json', result)
    write(output / 'ANALYSIS_COMPLETE.json', {'sha256': {name: S.sha(output / name)
        for name in ('REPORT.json', 'MEMBERS.json', 'RESULT.json')}})
    print({'actual_exit': 0, 'seconds': result['seconds'], 'accepted': report['accepted_count'],
        'holds': report['holds'], 'faults': report['faults']}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
