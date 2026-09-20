"""collector採録だけのprefix差分を、同runの実採否関数へ結び付ける。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import CodeType
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PROBE = ROOT.parent / 'g2_collector_publication_causality_2026-09-09_v1/probe.py'
BASE = PROJECT / 'scripts/diagnose_video38_confirmed_collapse_v1.py'
COLLECTOR = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/scripts/collect_boards_lean.py'
REFERENCE = ROOT.parent / 'video38_chigiri_completion_live_2026-09-08_v1'
KEY, C6 = 'publication_prefix_actual_collector', 30272
RECEIPT, REPLAY = 'PUBLICATION_PREFIX_RECEIPT.json', 'PUBLICATION_PREFIX_REPLAY.json'
REQUIRED = frozenset((RECEIPT, REPLAY))
CORE = ('frame_idx', 'side', 'board', 'board_provenance', 'chain_trigger_sec', 'chain_mechanism')
UNMEASURED = frozenset(('stable_persistence_confidence',))
SNAPSHOT_KEYS = frozenset((*CORE, 'kind', 'time_sec')) | UNMEASURED
FUNCTIONS = ('_process_side_lean', '_should_emit', '_update_game_boundary', '_update_move_scheduler',
             '_move_window_candidate_ok', '_is_physics_violation_persistent')
FIXED = {PROBE: 'f1ddeea5a65ef54bec23ed241790c28632bffa390f02d578868022731021b2fb',
    BASE: 'b20525d5b7423a34125a45b11f76aea68d0942276109463007dfd991e46fa144',
    COLLECTOR: '672963055a9fffa66531411be0a51742ee57c35de481652a77d02158ce24bee8',
    REFERENCE / 'COMPLETE': '373455e3131db33c8c7c5afb9f1497f39f47c1a3974575fd01326b410b8064b3'}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'prefix_fixed_source_changed')
    complete = read(REFERENCE / 'COMPLETE')
    reference = {str(REFERENCE / n): complete['sha256'][n] for n in ('frames.jsonl', 'PLAN.json')}
    require(all(sha(Path(p)) == h for p, h in reference.items()), 'prefix_reference_changed')
    return {str(p): h for p, h in FIXED.items()} | reference | {str(ROOT / n): sha(ROOT / n)
        for n in ('proof.py', 'finish.py', 'PLAN.md')}


def collector_identity(collector: Any) -> None:
    require(Path(collector.__file__).resolve() == COLLECTOR, 'prefix_collector_path')
    codes = {n.co_name: n for n in compile(COLLECTOR.read_bytes(), str(COLLECTOR), 'exec').co_consts
        if isinstance(n, CodeType)}
    require(all(getattr(collector, n).__code__ == codes[n] for n in FUNCTIONS), 'prefix_collector_function')


def load_probe(output: Path) -> Any:
    alias = '_bounded_collector_prefix_probe'
    require(alias not in sys.modules, 'prefix_probe_reentry')
    spec = importlib.util.spec_from_file_location(alias, PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[alias]
    module.NEW = output
    return module


def prefix_lines(path: Path) -> Iterator[bytes]:
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row['frame_idx'] < C6 and row['kind'] != 'collector_snapshot':
                yield line


def report_contract(output: Path, report: dict[str, Any]) -> None:
    require(type(report['first_c6']) is int and report['first_c6'] == C6, 'prefix_c6_changed')
    streams = report['streams']
    require(len(streams) == 4 and 'frames.jsonl' in streams, 'prefix_stream_set')
    require(all(r['pre_mutation_prefix_bit_exact'] is True for n, r in streams.items()
        if n != 'frames.jsonl'), 'prefix_other_stream_changed')
    require(all(r['equal'] is True and r['count_ok'] is True and r['structure_ok'] is True
        for r in report['invariants'].values()) and set(report['invariants']) == {'raw', 'next'}, 'prefix_raw_next')
    old, new = list(prefix_lines(REFERENCE / 'frames.jsonl')), list(prefix_lines(output / 'frames.jsonl'))
    require(old == new and bool(new), 'prefix_noncollector_changed')
    require(read(output / 'PLAN.json')['actual_collector_kwargs'] ==
        read(REFERENCE / 'PLAN.json')['actual_collector_kwargs'], 'prefix_actual_flags_changed')


def core(rows: list[Any], *, prefix: bool = False) -> list[Any]:
    return [{key: r[key] for key in CORE} for r in rows if not prefix or r['frame_idx'] < C6]


def snapshot_contract(actual: list[Any], old: list[Any]) -> None:
    """未検証列を一列に限定し、未知の列を丸ごと比較から落とさない。"""
    prior = {(r['frame_idx'], r['side']): r for r in old if r['frame_idx'] < C6}
    for row in actual:
        require(set(row) == SNAPSHOT_KEYS and row['kind'] == 'collector_snapshot', 'snapshot_schema_changed')
        require(type(row['time_sec']) is float and row['time_sec'] == row['frame_idx'] / 60, 'snapshot_time_changed')
        tag = row['stable_persistence_confidence']
        require(tag is None or type(tag) is bool, 'snapshot_stable_tag_type')
        previous = prior.get((row['frame_idx'], row['side']))
        if previous is not None:
            require(tag == previous['stable_persistence_confidence'], 'common_prefix_stable_tag_changed')


def replay(output: Path, collector: Any) -> dict[str, Any]:
    collector_identity(collector)
    probe = load_probe(output)
    base = probe.load_base()
    sides, actual = probe.frame_rows(output)
    _, old = probe.frame_rows(REFERENCE)
    snapshot_contract(actual, old)
    rows = probe.saved_returns(sides)
    lanes = {name: probe.replay(base, collector, rows, sides, name, True) for name in ('upstream', 'downstream')}
    require(core(lanes['downstream']['rows']) == core(actual), 'prefix_downstream_actual_mismatch')
    require(core(lanes['upstream']['rows'], prefix=True) == core(old, prefix=True), 'prefix_upstream_reference_mismatch')
    collector_identity(collector)
    return {'lanes': lanes, 'actual_snapshots': actual, 'old_prefix_snapshots': [r for r in old if r['frame_idx'] < C6],
        'updates': len(rows), 'downstream_core_equal': True, 'upstream_prefix_core_equal': True,
        'full_metadata_reproduced': False, 'quality_gate_clear': False}


def inputs(output: Path) -> dict[str, str]:
    names = ('PLAN.json', 'frames.jsonl', 'provisional_context.jsonl', 'consumer_publication.jsonl',
        'PRIVATE_CONSUMER_STATUS.json', 'PRIVATE_CONSUMER_RECEIPT.json', 'PROVISIONAL_CONTEXT_RECEIPT.json')
    return {str(output / n): sha(output / n) for n in names}


def prove(output: Path, report: dict[str, Any], state: dict[str, Any]) -> None:
    before = guards() | inputs(output)
    report_contract(output, report)
    result = replay(output, state[KEY])
    require(before == {p: sha(Path(p)) for p in before}, 'prefix_replay_input_changed')
    write(output / REPLAY, result)
    write(output / RECEIPT, {'guards': before, 'sha256': {REPLAY: sha(output / REPLAY)},
        'original_report_digest': digest(report), 'output': output.resolve().as_posix(),
        'old_prefix_bit_exact': report['streams']['frames.jsonl']['pre_mutation_prefix_bit_exact'],
        'actual_collector_object_used': True, 'source_injected_or_reseeded': False,
        'shared_game_metadata': 'NOT_MEASURED', 'stable_image_tags': 'NOT_MEASURED',
        'unmeasured_snapshot_fields': sorted(UNMEASURED),
        'quality_gate_clear': False, 'production_permission': False})
    validate(output, report, state)


def verify(output: Path) -> None:
    receipt = read(output / RECEIPT)
    require(receipt['output'] == output.resolve().as_posix(), 'prefix_output_identity')
    require(all(sha(Path(p)) == h for p, h in receipt['guards'].items()), 'prefix_saved_input_changed')
    require(set(receipt['sha256']) == {REPLAY} and sha(output / REPLAY) == receipt['sha256'][REPLAY], 'prefix_saved_replay_changed')
    require(all(receipt[key] is False for key in ('quality_gate_clear', 'production_permission',
        'source_injected_or_reseeded')) and receipt['actual_collector_object_used'] is True, 'prefix_saved_permissions')


def validate(output: Path, report: dict[str, Any], state: dict[str, Any]) -> None:
    verify(output)
    collector_identity(state[KEY])
    require(read(output / RECEIPT)['original_report_digest'] == digest(report), 'prefix_report_changed')
