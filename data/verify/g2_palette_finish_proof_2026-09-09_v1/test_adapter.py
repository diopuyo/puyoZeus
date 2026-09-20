"""人工再束縛保存正例と必要な拒否境界だけ。再認識/原run救済なし。"""
from __future__ import annotations
import contextlib
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import adapter as A
import cpu_fixture as C

OUTPUT = Path(os.environ['PALETTE_PROOF_CPU_OUTPUT'])


@pytest.fixture(scope='module')
def completed() -> Any:
    proof, dep = A.load(A.PROOF, 'test_proof'), A.dependencies()
    output = OUTPUT / 'fixture'
    before = C.prepare(output, proof, dep)
    report = C.report(proof, dep, output)
    base = proof.load_probe(output).load_base()
    collector = base.load_collector()
    yield SimpleNamespace(proof=proof, dep=dep, output=output, before=before,
        report=report, base=base, collector=collector)
    assert before == {p: A.sha(Path(p)) for p in before}
    assert not (C.FAILED / 'COMPLETE').exists()


def test_actual_saved_proof_and_original_verify(completed: Any) -> None:
    c = completed
    with pytest.raises(ValueError, match='prefix_noncollector_changed'):
        c.proof.report_contract(c.output, c.report)
    original = {n: getattr(c.proof, n) for n in ('prove', 'verify', 'guards', 'inputs', 'digest', 'read', 'sha')}
    with contextlib.ExitStack() as stack:
        A.install(stack, c.proof, enabled=True)
        c.proof.prove(c.output, c.report, {c.proof.KEY: c.collector})
        assert all(getattr(c.proof, n) is f for n, f in original.items() if n != 'prove')
    assert all(getattr(c.proof, n) is f for n, f in original.items())
    c.proof.verify(c.output)
    A.verify(c.output)
    result = A.read(c.output / c.proof.REPLAY)
    assert result['palette_compatibility']['veto_cells'] == 926
    assert len(result['palette_compatibility']['metadata_appends']) == 228
    assert A.read(c.output / c.proof.RECEIPT)['old_prefix_bit_exact'] is False
    C.write(OUTPUT / 'ACTUAL_PROOF_SUMMARY.json', {'veto_cells': 926, 'metadata_appends': 228,
        'all_21_columns': result['palette_compatibility']['metadata_columns'],
        'actual_upstream_prefix_equal': result['upstream_prefix_core_equal'],
        'virtual_prefix_equal': True, 'old_failed_exit': 1, 'artificial_rebound_fixture': True,
        'quality_gate_clear': False, 'original_proof_verify_passed': True})


@pytest.mark.parametrize('change', ('score', 'cnn', 'frame', 'unknown_column'))
def test_unrelated_change_rejected(completed: Any, change: str) -> None:
    c = completed
    veto = next(A.lines(c.output / 'palette_evidence_veto.jsonl'))
    key = veto['frame_idx'], veto['side']
    new = next(r for r in A.lines(c.output / 'frames.jsonl') if r['kind'] == 'frame_side'
        and (r['frame_idx'], r['side']) == key)
    old = copy.deepcopy(new)
    old['confirmed'] = c.dep.e.inverse(c.base, new['confirmed'], veto)
    bad = copy.deepcopy(new)
    if change == 'unknown_column':
        bad[change] = 1
    elif change == 'cnn':
        bad['cnn']['grid'][12][0] = (bad['cnn']['grid'][12][0] + 1) % 6
    elif change == 'frame':
        bad['frame_idx'] += 2
    else:
        bad['score'] += 1
    with pytest.raises(ValueError, match='noncolor_or_unbound'):
        c.dep.e.frame_pair(c.base, old, bad, veto)


@pytest.mark.parametrize('field,value', [('side', '2P'), ('frame_idx', True), ('time_sec', 0.0), ('run_id', 'foreign')])
def test_clock_and_source(completed: Any, field: str, value: Any) -> None:
    row = next(A.lines(completed.output / 'palette_evidence_veto.jsonl'))
    row[field] = value
    if field == 'side':
        row['side'] = '3P'
    with pytest.raises(ValueError):
        completed.dep.e.clock(row, completed.output)


def test_default_off_exception_and_restore(completed: Any) -> None:
    c, marker = completed, RuntimeError('original-marker')
    original = c.proof.prove
    with contextlib.ExitStack() as stack:
        A.install(stack, c.proof)
        assert c.proof.prove is original
    with pytest.raises(RuntimeError) as caught:
        with contextlib.ExitStack() as stack:
            A.install(stack, c.proof, enabled=True)
            raise marker
    assert caught.value is marker and c.proof.prove is original
    with contextlib.ExitStack() as stack:
        A.install(stack, c.proof, enabled=True)
        with pytest.raises(ValueError, match='palette_proof_function:prove'):
            A.install(stack, c.proof, enabled=True)


def test_changed_digest_rejected(completed: Any) -> None:
    c = completed
    report = copy.deepcopy(c.report)
    report['first_c6'] += 2
    with pytest.raises(ValueError, match='prefix_report_changed'):
        c.proof.validate(c.output, report, {c.proof.KEY: c.collector})


def test_report_raw_next_and_other_stream(completed: Any) -> None:
    c = completed
    for kind in ('count', 'other_stream', 'input_sha', 'false_to_true'):
        value = copy.deepcopy(c.report)
        if kind == 'count':
            value['invariants']['raw']['new'] = True
        elif kind == 'other_stream':
            value['streams']['current_entry.jsonl']['pre_mutation_prefix_bit_exact'] = False
        elif kind == 'input_sha':
            value['streams']['frames.jsonl']['input_sha256'][str(c.output / 'frames.jsonl')] = '0' * 64
        else:
            value['streams']['frames.jsonl']['pre_mutation_prefix_bit_exact'] = True
        with pytest.raises(ValueError):
            c.dep.e.report_values(c.proof, c.output, value)


def test_duplicate_and_nonfinite_json() -> None:
    for value in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError):
            A.parse(value)


def test_saved_journal_code_rejected(completed: Any) -> None:
    j = completed.dep.journal
    row = next(r for r in A.lines(completed.output / 'atomic_journal.jsonl') if r['kind'] == 'step')
    j.verify_stages(row, j.anchors())
    row['code_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='journal_saved_code'):
        j.verify_stages(row, j.anchors())


def test_real_history_save_and_finalizer_once(completed: Any, monkeypatch: Any) -> None:
    import history_cpu as H
    H.execute(completed, monkeypatch)
