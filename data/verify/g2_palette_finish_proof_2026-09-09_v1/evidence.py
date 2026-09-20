"""同時計の留保・上流・journalだけで色差を限定する。"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Iterator

A: Any = None
STREAMS = frozenset(('frames.jsonl', 'current_entry.jsonl', 'resolved_grace_guard.jsonl', 'chigiri_completion.jsonl'))


def clock(row: Any, output: Path) -> tuple[int, str]:
    frame, side = row['frame_idx'], row['side']
    A.require(type(frame) is int and A.FIRST <= frame <= A.LAST and (frame - A.FIRST) % A.STRIDE == 0
        and side in A.SIDES and type(row['time_sec']) in (float, int)
        and A.equal(row['time_sec'], frame / A.FPS), 'palette_proof_clock')
    A.require(row['source_id'] == A.SOURCE_ID and row['run_id'] == output.resolve().as_posix(), 'palette_proof_source_run')
    return frame, side


def grid(value: Any) -> Any:
    if value is None:
        return None
    result = value['grid']
    A.require(type(result) is list and len(result) == 13 and all(type(r) is list and len(r) == 6 for r in result)
        and all(type(c) is int and c in (0, 1, 2, 3, 4, 5, 9, 10) for r in result for c in r), 'palette_proof_grid')
    return result


def palette_rows(dep: Any, output: Path) -> dict[Any, Any]:
    dep.palette.verify(output)
    result: dict[Any, Any] = {}
    for row in A.lines(output / dep.palette.SIDECAR):
        key = clock(row, output)
        A.require(key not in result and row['cells'], 'palette_proof_duplicate_or_empty_veto')
        cells = {(c['row'], c['col']) for c in row['cells']}
        A.require(len(cells) == len(row['cells']), 'palette_proof_duplicate_cell')
        result[key] = row
    return result


def frame_lines(path: Path, *, prefix: bool = False) -> Iterator[Any]:
    with path.open('rb') as stream:
        for line in stream:
            row = A.parse(line)
            if row['kind'] != 'collector_snapshot' and (not prefix or row['frame_idx'] < A.C6):
                yield line, row


def inverse(base: Any, value: Any, veto: Any) -> Any:
    result = copy.deepcopy(grid(value))
    for cell in veto['cells']:
        if cell['vetoed']:
            r, c = cell['row'], cell['col']
            A.require(result is not None and result[r][c] == cell['original'], 'palette_proof_lost_veto')
            result[r][c] = cell['replacement']
    return None if result is None else base.board_value(result)


def frame_pair(base: Any, old: Any, new: Any, veto: Any) -> bool:
    if A.equal(old, new):
        A.require(veto is None or not any(c['vetoed'] for c in veto['cells']), 'palette_proof_veto_not_reflected')
        return False
    A.require(old['kind'] == new['kind'] == 'frame_side' and veto is not None, 'palette_proof_unexplained_row')
    hypothetical = dict(new)
    hypothetical['confirmed'] = inverse(base, new['confirmed'], veto)
    A.require(A.equal(old, hypothetical), 'palette_proof_noncolor_or_unbound_change')
    return True


def compare_frames(base: Any, reference: Path, output: Path, vetoes: Any, *, prefix: bool = False) -> list[Any]:
    changes = []
    for (text_a, old), (text_b, new) in zip(frame_lines(reference / 'frames.jsonl', prefix=prefix),
            frame_lines(output / 'frames.jsonl', prefix=prefix), strict=True):
        A.require((old['kind'], old['frame_idx'], old.get('side')) ==
            (new['kind'], new['frame_idx'], new.get('side')), 'palette_proof_row_order')
        if new['kind'] != 'frame_side':
            A.require(text_a == text_b, 'palette_proof_nonframe_text_changed')
        elif frame_pair(base, old, new, vetoes.get((new['frame_idx'], new['side']))):
            changes.append((new['frame_idx'], new['side']))
    return changes


def report_values(proof: Any, output: Path, report: Any) -> None:
    A.require(type(report['first_c6']) is int and report['first_c6'] == A.C6, 'palette_proof_first_c6')
    A.require(set(report['streams']) == STREAMS and set(report['invariants']) == {'raw', 'next'}, 'palette_proof_streams')
    for name, row in report['streams'].items():
        paths = (proof.REFERENCE / name, output / name)
        A.require(row['input_sha256'] == {str(p): A.sha(p) for p in paths}, 'palette_proof_report_input_sha')
        A.require(type(row['pre_mutation_prefix_bit_exact']) is bool, 'palette_proof_prefix_type')
        if name != 'frames.jsonl':
            A.require(row['pre_mutation_prefix_bit_exact'] is True, 'palette_proof_other_prefix')
            streams = [frame_lines(p, prefix=True) for p in paths]
            A.require([x[0] for x in streams[0]] == [x[0] for x in streams[1]], 'palette_proof_other_text')
    for name, count in (('raw', 7248), ('next', 2100)):
        row = report['invariants'][name]
        A.require(all(row[k] is True for k in ('equal', 'count_ok', 'structure_ok'))
            and type(row['old']) is int and type(row['new']) is int and row['old'] == row['new'] == count,
            'palette_proof_raw_next_coverage')
    A.require(A.equal(A.read(output / 'PLAN.json')['actual_collector_kwargs'],
        A.read(proof.REFERENCE / 'PLAN.json')['actual_collector_kwargs']), 'palette_proof_actual_flags')
    prefixes = []
    for path in (proof.REFERENCE / 'frames.jsonl', output / 'frames.jsonl'):
        with path.open('rb') as stream:
            prefixes.append([line for line in stream if A.parse(line)['frame_idx'] < A.C6])
    A.require(report['streams']['frames.jsonl']['pre_mutation_prefix_bit_exact'] is (prefixes[0] == prefixes[1]),
        'palette_proof_actual_prefix_flag')


def journal_rows(dep: Any, output: Path, vetoes: Any) -> tuple[dict[Any, Any], int]:
    found, calls, steps, seen = {}, 0, 0, set()
    targets = dep.journal.anchors()
    for index, row in enumerate(A.lines(output / 'atomic_journal.jsonl')):
        key = clock(row, output)
        A.require(type(row['row_index']) is int and row['row_index'] == index and row['status'] == 'returned'
            and row['token'] not in seen and row['schema'] == dep.journal.SCHEMA
            and all(row[k] is False for k in dep.journal.PERMISSIONS), 'palette_proof_journal_status_order')
        seen.add(row['token'])
        if row['kind'] != 'step':
            A.require(row['kind'] == 'enqueue', 'palette_proof_journal_kind')
            dep.journal.verify_enqueue(row)
            continue
        dep.journal.verify_stages(row, targets)
        expected = (A.FIRST + steps // 2 * A.STRIDE, A.SIDES[steps % 2])
        A.require(key == expected and row['token'] == 'step:' + str(steps), 'palette_proof_journal_coverage')
        steps += 1
        stages = [e['stage'] for e in row['events']]
        if 'next_validation_before' in stages:
            A.require(stages.count('next_validation_before') == stages.count('next_validation_after') == 1,
                'palette_proof_journal_validator_count')
            if key in vetoes:
                A.require(vetoes[key]['call_index'] == calls, 'palette_proof_veto_call_index')
                found[key] = row
            calls += 1
    A.require(steps == 7248 and set(found) == set(vetoes), 'palette_proof_journal_scope_missing')
    return found, calls


def bind_journal(row: Any, veto: Any, final: Any, cnn: Any) -> None:
    stages = {e['stage']: e for e in row['events']}
    before, after = stages['next_validation_before'], stages['next_validation_after']
    A.require(before['state'] == after['state'] == 'STABLE', 'palette_proof_nonstable_veto')
    target = grid(final)
    A.require(grid(after['boards']['published_confirmed']) == target, 'palette_proof_next_final_mismatch')
    for name in ('persistence_after', 'glow_after', 'answer_check_after', 'publication_after'):
        A.require(name in stages and grid(stages[name]['boards']['published_confirmed']) == target,
            'palette_proof_later_writer_changed:' + name)
    for cell in veto['cells']:
        r, c = cell['row'], cell['col']
        A.require(grid(before['confirmed'])[r][c] == cell['original'], 'palette_proof_original_input_cell')
        if cell['vetoed']:
            A.require(grid(cnn)[r][c] == grid(before['boards']['cnn_board'])[r][c] == cell['original'],
                'palette_proof_cnn_join')
    A.require(before['accounting'] == after['accounting'] and before['confirmed'] == after['confirmed'],
        'palette_proof_context_or_account_writer')


def bind_context(output: Path, vetoes: Any, journals: Any, sides: Any) -> None:
    found = set()
    streams = (A.lines(output / 'provisional_context.jsonl'), A.lines(output / 'consumer_publication.jsonl'))
    for index, (ctx, pub) in enumerate(zip(*streams, strict=True)):
        frame = A.FIRST + index * A.STRIDE
        A.require(type(ctx['frame_idx']) is int and type(pub['frame_idx']) is int
            and ctx['frame_idx'] == pub['frame_idx'] == frame and A.equal(ctx['time_sec'], frame / A.FPS)
            and A.equal(pub['time_sec'], frame / A.FPS) and pub['context_digest'] == A.digest(ctx), 'palette_proof_context_digest_clock')
        A.require(ctx['failures'] == [] and ctx['update']['returned'] is True, 'palette_proof_context_failure')
        for side in A.SIDES:
            key = frame, side
            final = ctx['sides'][side]['final']
            A.require(grid(final['confirmed']) == grid(sides[key]['confirmed']), 'palette_proof_frame_context_grid')
            if key in vetoes:
                pb = ctx['sides'][side]['pb']
                A.require(clock(pb, output) == key, 'palette_proof_pb_scope')
                bind_journal(journals[key], vetoes[key], final['confirmed'], final['cnn'])
                A.require(A.equal(journals[key]['generation_after'], ctx['generation']['after']['value'][side]),
                    'palette_proof_generation_join')
                found.add(key)
    A.require(index + 1 == 3624 and found == set(vetoes), 'palette_proof_context_coverage')


def audit(proof: Any, dep: Any, output: Path, report: Any) -> dict[str, Any]:
    report_values(proof, output, report)
    plan = A.read(output / 'PLAN.json')
    A.require(plan['input_and_code_sha256'][plan['video_path']] == A.SOURCE_ID.removeprefix('sha256:'),
        'palette_proof_video_source')
    vetoes = palette_rows(dep, output)
    probe = proof.load_probe(output)
    base = probe.load_base()
    sides, _ = probe.frame_rows(output)
    whole = compare_frames(base, A.BASELINE, output, vetoes)
    prefix = compare_frames(base, proof.REFERENCE, output, vetoes, prefix=True)
    journals, calls = journal_rows(dep, output, vetoes)
    A.require(calls == A.read(output / dep.palette.RECEIPT)['calls'], 'palette_proof_palette_calls')
    bind_context(output, vetoes, journals, sides)
    return {'output': str(output.resolve()), 'vetoes': vetoes, 'changed_frame_sides': whole,
        'prefix_changed_frame_sides': prefix, 'validator_calls': calls}
