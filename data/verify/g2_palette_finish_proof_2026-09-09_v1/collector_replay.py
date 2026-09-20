"""元採否と元appendを再生し、仮想対照を実上流と混同しない。"""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

A: Any = None


def metadata_functions(observer: Any) -> dict[str, Any]:
    namespace = {'Any': Any, 'O': observer, 'L': SimpleNamespace(F=SimpleNamespace(require=A.require))}
    return A.source_functions(A.REPLAY_SOURCE, ('decode', 'replay'), namespace)


def metadata_join(dep: Any, collector: Any, output: Path, sides: Any, rows: Any, actual: Any) -> list[Any]:
    O, functions = dep.metadata, metadata_functions(dep.metadata)
    status = A.read(output / O.STATUS)
    A.require(status['closed'] is True and status['errors'] == [] and status['rows'] == 7248
        and status['label_columns_read'] is False and status['quality_gate_clear'] is False, 'palette_proof_metadata_status')
    records, count = [], 0
    for index, item in enumerate(A.lines(output / O.ENTRIES)):
        frame, side = A.FIRST + index // 2 * A.STRIDE, A.SIDES[index % 2]
        count = O.validate_row(item, frame, side, count)
        value = rows[index // 2]['downstream'][side]
        args = item['arguments']
        decode = functions['decode']
        board = None if args['board'] is None else decode(args['board']).tolist()
        A.require(board == dep.e.grid(value['confirmed']) and args['bstate']['value'] == value['state_value']
            and args['score'] == sides[frame, side]['score'], 'palette_proof_metadata_publication_join')
        for name in ('next_pair', 'dnext_pair'):
            pair = decode(args[name])
            A.require((None if pair is None else list(pair)) == value[name], 'palette_proof_metadata_queue')
        for append in item['appends']:
            replay = functions['replay'](collector, append)
            A.require(not replay['differences'], 'palette_proof_metadata_21_mismatch')
            stored = append['stored_nonlabel_row']
            inputs = decode(append['arguments'])
            snapshot = actual[len(records)]
            A.require((frame, side) == (snapshot['frame_idx'], snapshot['side'])
                and decode(stored['grids']).tolist() == snapshot['board']['grid']
                and stored['board_provenances'] == snapshot['board_provenance']
                and inputs['chain_trigger_sec'] == snapshot['chain_trigger_sec']
                and inputs['mechanism'] == snapshot['chain_mechanism']
                and A.equal(args['stable_persistence_confidence'], snapshot['stable_persistence_confidence']),
                'palette_proof_metadata_snapshot_join')
            records.append(replay)
    A.require(index + 1 == 7248 and count == len(records) == len(actual), 'palette_proof_metadata_coverage')
    return records


def virtual_prefix(dep: Any, base: Any, rows: Any, vetoes: Any) -> list[Any]:
    result = copy.deepcopy([r for r in rows if r['frame'] < A.C6])
    for row in result:
        for side in A.SIDES:
            veto = vetoes.get((row['frame'], side))
            if veto is not None:
                row['upstream'][side]['confirmed'] = dep.e.inverse(base, row['upstream'][side]['confirmed'], veto)
    return result


def replay(proof: Any, dep: Any, output: Path, collector: Any, evidence: Any) -> dict[str, Any]:
    proof.collector_identity(collector)
    probe = proof.load_probe(output)
    base = probe.load_base()
    sides, actual = probe.frame_rows(output)
    _, old = probe.frame_rows(proof.REFERENCE)
    proof.snapshot_contract(actual, old)
    rows = probe.saved_returns(sides)
    lanes = {n: probe.replay(base, collector, rows, sides, n, True) for n in ('upstream', 'downstream')}
    A.require(proof.core(lanes['downstream']['rows']) == proof.core(actual), 'palette_proof_downstream_actual')
    virtual = virtual_prefix(dep, base, rows, evidence['vetoes'])
    virtual_lane = probe.replay(base, collector, virtual, sides, 'upstream', True)
    A.require(proof.core(virtual_lane['rows']) == proof.core(old, prefix=True), 'palette_proof_virtual_prefix')
    metadata = metadata_join(dep, collector, output, sides, rows, actual)
    proof.collector_identity(collector)
    compatibility = {'same_call_veto_bound': True, 'virtual_lane_not_physical_truth': True,
        'virtual_prefix_core_equal': True, 'metadata_21_columns_equal': True,
        'changed_frame_sides': evidence['changed_frame_sides'], 'prefix_changes': evidence['prefix_changed_frame_sides'],
        'veto_cells': sum(c['vetoed'] for v in evidence['vetoes'].values() for c in v['cells']),
        'metadata_columns': sorted(dep.metadata.STORED_KEYS), 'metadata_appends': metadata,
        'virtual_prefix_lane': virtual_lane, 'quality_gate_clear': False,
        'accounting_permission': False, 'physical_truth_certified': False}
    return {'lanes': lanes, 'actual_snapshots': actual, 'old_prefix_snapshots': [r for r in old if r['frame_idx'] < A.C6],
        'updates': len(rows), 'downstream_core_equal': True,
        'upstream_prefix_core_equal': proof.core(lanes['upstream']['rows'], prefix=True) == proof.core(old, prefix=True),
        'full_metadata_reproduced': False, 'quality_gate_clear': False, 'palette_compatibility': compatibility}
