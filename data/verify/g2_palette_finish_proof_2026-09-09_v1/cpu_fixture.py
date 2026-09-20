"""失敗原本は不変。別出力の人工run識別子へ保存値を再束縛するCPU専用準備。"""
from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
from typing import Any
import adapter as A

FAILED = A.VERIFY / 'video38_palette_veto_live_2026-09-09_v1'


def write(path: Path, value: Any) -> None:
    import json
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def copy_stream(source: Path, target: Path) -> None:
    old, new = str(source.parent.resolve()), str(target.parent.resolve())
    with source.open() as src, target.open('x') as dst:
        for line in src:
            dst.write(line.replace(old, new))


def publication(output: Path) -> None:
    import json
    path = output / 'consumer_publication.jsonl'
    sources = (A.lines(output / 'provisional_context.jsonl'), A.lines(FAILED / path.name))
    with path.open('x') as dst:
        for ctx, pub in zip(*sources, strict=True):
            pub['context_digest'] = A.digest(ctx)
            dst.write(json.dumps(pub, ensure_ascii=False, allow_nan=False) + '\n')


def palette_receipt(output: Path, dep: Any) -> None:
    rows = list(A.lines(output / dep.palette.SIDECAR))
    calls = sum(1 for r in A.lines(output / 'atomic_journal.jsonl') if r['kind'] == 'step'
        for e in r['events'] if e['stage'] == 'next_validation_before')
    reasons = Counter(c['reason'] for r in rows for c in r['cells'])
    write(output / dep.palette.RECEIPT, {'closed': True, 'calls': calls, 'rows': len(rows),
        'vetoed_cells': sum(c['vetoed'] for r in rows for c in r['cells']), 'reasons': dict(reasons),
        'guards': dep.palette.guards(), 'sha256': {dep.palette.SIDECAR: A.sha(output / dep.palette.SIDECAR)},
        'quality_gate_clear': False, 'physical_palette_certified': False,
        'artificial_cpu_receipt_reconstructed_from_saved_calls': True})


def prepare(output: Path, proof: Any, dep: Any) -> dict[str, str]:
    A.require(not (FAILED / 'COMPLETE').exists() and A.read(FAILED / 'CHILD_EXIT.json')['child_exit_code'] == 1,
        'old_failure_must_remain')
    output.mkdir(exist_ok=False)
    names = set(A.EXTRA) - {'PALETTE_EVIDENCE_VETO.json'}
    names |= {'PLAN.json', 'frames.jsonl', 'provisional_context.jsonl', 'consumer_publication.jsonl',
        'PRIVATE_CONSUMER_STATUS.json', 'PRIVATE_CONSUMER_RECEIPT.json', 'PROVISIONAL_CONTEXT_RECEIPT.json',
        'current_entry.jsonl', 'resolved_grace_guard.jsonl', 'chigiri_completion.jsonl'}
    before = {str(FAILED / n): A.sha(FAILED / n) for n in names}
    for name in names - {'PLAN.json', 'consumer_publication.jsonl'}:
        if name in ('provisional_context.jsonl', 'atomic_journal.jsonl', 'palette_evidence_veto.jsonl'):
            copy_stream(FAILED / name, output / name)
        else:
            os.link(FAILED / name, output / name)
    publication(output)
    plan = A.read(FAILED / 'PLAN.json')
    plan['input_and_code_sha256'].update(A.guards())
    write(output / 'PLAN.json', plan)
    palette_receipt(output, dep)
    write(output / 'CPU_FIXTURE.json', {'original_failed_run': str(FAILED), 'original_exit': 1,
        'artificial_rebound_run_id': str(output), 'live_success_or_quality_permission': False,
        'source_values_preserved_except_identity_and_dependent_digest': True, 'source_sha256': before})
    return before


def report(proof: Any, dep: Any, output: Path) -> dict[str, Any]:
    streams = {}
    for name in dep.e.STREAMS:
        paths = (proof.REFERENCE / name, output / name)
        prefix = []
        for path in paths:
            with path.open('rb') as src:
                prefix.append([line for line in src if A.parse(line)['frame_idx'] < A.C6])
        streams[name] = {'input_sha256': {str(p): A.sha(p) for p in paths},
            'pre_mutation_prefix_bit_exact': prefix[0] == prefix[1]}
    return {'streams': streams, 'invariants': {k: {'equal': True, 'count_ok': True, 'structure_ok': True,
        'old': n, 'new': n} for k, n in (('raw', 7248), ('next', 2100))}, 'first_c6': A.C6,
        'post_intervention_differences_require_review': True, 'quality_gate_clear': False}
