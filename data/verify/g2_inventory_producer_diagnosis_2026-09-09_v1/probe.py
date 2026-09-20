"""同run在庫・着手の機械証拠だけを抽出する。認証や会計writerは呼ばない。

資産確認: SplitOwnerの基点/placement契約と既存ACCOUNTING_NEXTを再利用。
配置は同run snapshot puyo_core_bridge.enumerate_placements(filter_dead=False)。
左右同高さ限定inferrer、推論再実行、新手ID、旧Counterの基点化は使用しない。
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

UNIT = Path(__file__).resolve().parent
PROJECT = UNIT.parents[2]
LIVE = PROJECT / 'data/verify/video38_provisional_context_live_2026-09-09_v1'
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'
SIDES = ('1P', '2P')
COLORS = frozenset(range(1, 6))
VALID = COLORS | {0, 9}
ROWS, COLS, STRIDE = 13, 6, 2
Grid = tuple[tuple[int, ...], ...]
LIVE_FILES = ('COMPLETE', 'PLAN.json', 'CHILD_EXIT.json', 'frames.jsonl',
              'provisional_context.jsonl', 'PROVISIONAL_CONTEXT_RECEIPT.json',
              'PROVISIONAL_CONTEXT_STATUS.json')
COMPLETE_SHA = '004b128fc0c99a61ff86dd9a336f3169e9315b9306b7c78092c004ccd8746c33'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def grid(value: Any) -> Grid | None:
    value = value.get('grid') if isinstance(value, dict) else value
    if not isinstance(value, (list, tuple)) or len(value) != ROWS:
        return None
    if any(not isinstance(row, (list, tuple)) or len(row) != COLS for row in value):
        return None
    if any(type(cell) is not int or cell not in VALID for row in value for cell in row):
        return None
    return tuple(tuple(row) for row in value)


def grid_sha(value: Grid | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def pointmass(pb: dict[str, Any]) -> Grid | None:
    if pb.get('present') is not True or pb.get('type_valid') is not True or pb.get('errors'):
        return None
    cells = pb.get('cells')
    if not isinstance(cells, list) or len(cells) != ROWS:
        return None
    result = []
    for row in cells:
        if not isinstance(row, list) or len(row) != COLS:
            return None
        out = []
        for cell in row:
            if not isinstance(cell, list) or not cell:
                return None
            if any(not isinstance(x, list) or len(x) != 2 for x in cell):
                return None
            if any(type(c) is not int or c not in VALID or type(p) not in (int, float)
                   or not math.isfinite(p) or p < 0 for c, p in cell):
                return None
            if len({c for c, _ in cell}) != len(cell) or sum(p for _, p in cell) != 1:
                return None
            positive = [c for c, p in cell if p > 0]
            if len(positive) != 1:
                return None
            out.append(positive[0])
        result.append(out)
    return grid(result)


def controlled(row: dict[str, Any]) -> bool:
    up = row['update']
    fields = {'is_match_active': True, 'match_end_locked': False,
              'post_match_lockdown_active': False}
    return (row.get('failures') == [] and row.get('capture_status') == 'CAPTURED'
            and all(up.get(k) is v and up.get(k + '_observed') is True
                    for k, v in fields.items()))


def epoch(row: dict[str, Any], side: str, stage: str = 'after') -> tuple[Any, Any]:
    generation = row['generation'][stage]
    require(generation.get('observed') is True, 'generation_unobserved')
    value = generation['value'][side]
    require(value['side'] == side, 'generation_side')
    return value['reset_epoch'], value['action_revision']


def frame_index() -> tuple[dict[Any, Any], dict[str, Any]]:
    rows, kinds, samples = defaultdict(dict), Counter(), {}
    with (LIVE / 'frames.jsonl').open() as stream:
        for number, line in enumerate(stream, 1):
            value = json.loads(line)
            kind, frame, side = value['kind'], value['frame_idx'], value.get('side')
            kinds[kind] += 1
            samples.setdefault(kind, {'line': number, 'keys': list(value)})
            if frame < 0:
                continue
            if kind in ('frame_side', 'accounting_update'):
                require(kind not in rows[(frame, side)], 'duplicate_frame_side_kind')
                rows[(frame, side)][kind] = (number, value)
            elif kind not in ('read',):
                rows[(frame, None)].setdefault(kind, []).append((number, value))
    return dict(rows), {'kinds': dict(kinds), 'first_schema': samples}


def raw_at(frames: dict[Any, Any], frame: int, side: str) -> Grid | None:
    item = frames.get((frame, side), {}).get('frame_side')
    if item is None:
        return None
    value = item[1]
    capture = value.get('accounting_capture') or {}
    if value.get('raw_captured_this_frame') is not True or capture.get('captured_frame') != frame:
        return None
    return grid(capture.get('raw'))


def summarize(row: dict[str, Any], side: str, frames: dict[Any, Any]) -> dict[str, Any]:
    final, frame = row['sides'][side]['final'], row['frame_idx']
    confirmed = grid(final.get('confirmed'))
    raw = raw_at(frames, frame, side)
    pb = pointmass(final.get('probability') or {})
    fr = frames[(frame, side)]['frame_side']
    require(fr[1]['time_sec'] == row['time_sec'], 'frame_clock_join')
    require(grid(fr[1].get('confirmed')) == confirmed, 'frame_confirmed_join')
    account = frames[(frame, side)].get('accounting_update')
    next_rows = frames.get((frame, None), {})
    next_evidence = {kind: [dict(line=n, value=v) for n, v in next_rows.get(kind, [])
                            if v.get('side') in (None, side)]
                     for kind in ('next_enqueue_live_decision', 'next_enqueue_live_accounting')}
    return {'frame': frame, 'time_sec': row['time_sec'], 'side': side,
            'epoch': epoch(row, side), 'before_epoch': epoch(row, side, 'before'),
            'state': final['state'], 'control_ok': controlled(row),
            'grid': confirmed, 'grid_sha': grid_sha(confirmed), 'raw': raw,
            'raw_sha': grid_sha(raw), 'pb_pointmass': pb is not None,
            'pb_grid': pb, 'pb_sha': final.get('probability', {}).get('sha256'),
            'machine_current_match': confirmed is not None and confirmed == raw == pb,
            'next_pair': final.get('next_pair'), 'dnext_pair': final.get('dnext_pair'),
            'frame_line': fr[0], 'accounting_line': account[0] if account else None,
            'accounting': account[1] if account else None,
            'chain': fr[1].get('chain'), 'candidate_scope': row['candidate_scope_selected'],
            'next_evidence': next_evidence}


def difference(before: Grid | None, after: Grid | None) -> dict[str, Any]:
    if before is None or after is None:
        return {'kind': 'missing_grid', 'cells': [], 'pair': None}
    changes = [(r, c, before[r][c], after[r][c]) for r in range(ROWS)
               for c in range(COLS) if before[r][c] != after[r][c]]
    pair = [new for _, _, old, new in changes if old == 0 and new in COLORS]
    if not changes:
        kind = 'same_grid'
    elif any(9 in (old, new) for _, _, old, new in changes):
        kind = 'garbage_change'
    elif any(old != 0 for _, _, old, _ in changes):
        kind = 'baseline_cell_changed'
    elif len(pair) == 1:
        kind = 'plus_one_partial'
    elif len(pair) == 2:
        kind = 'two_color_addition'
    else:
        kind = 'other_increase'
    return {'kind': kind, 'cells': changes, 'pair': pair if len(pair) == 2 else None}


def placement_matches(before: Grid, after: Grid, pair: list[int]) -> list[dict[str, Any]]:
    core = importlib.import_module('src.puyo_core_bridge')
    board = core.Board()
    for r in range(ROWS):
        for c in range(COLS):
            board.set(r, c, before[r][c])
    hits = []
    for ordered in sorted(set(itertools.permutations(pair))):
        options = core.enumerate_placements(board, ordered, filter_dead=False)
        for col, rotation, placed in options:
            if grid(placed.to_dict()) == after:
                hits.append({'pair': ordered, 'col': col, 'rotation': rotation,
                             'enumerated_count': len(options)})
    require(grid(board.to_dict()) == before, 'enumerator_modified_input')
    return hits


def compact(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in ('raw', 'pb_grid')}


def cycle_result(start: dict[str, Any], middle: list[dict[str, Any]],
                 end: dict[str, Any]) -> dict[str, Any]:
    delta = difference(start['grid'], end['grid'])
    all_rows = [start] + middle + [end]
    fall_epochs = {tuple(r['epoch']) for r in middle + [end]}
    exact_epoch = len({tuple(r['epoch']) for r in all_rows}) == 1
    known_action = end['epoch'][1] is not None
    hits = placement_matches(start['grid'], end['grid'], delta['pair']) if delta['pair'] else []
    raw_diffs = [difference(start['grid'], row['raw']) for row in middle]
    reason = []
    for ok, text in ((delta['kind'] == 'two_color_addition', delta['kind']),
                     (bool(hits), 'no_22_placement_match'),
                     (len(fall_epochs) == 1 and known_action, 'fall_action_not_single'),
                     (all(r['control_ok'] for r in all_rows), 'control_hold'),
                     (start['machine_current_match'], 'baseline_not_machine_match'),
                     (end['machine_current_match'], 'end_not_machine_match')):
        if not ok:
            reason.append(text)
    return {'start': compact(start), 'end': compact(end),
            'fall_frames': [r['frame'] for r in middle],
            'fall_epochs': sorted(fall_epochs, key=str), 'all_same_epoch': exact_epoch,
            'delta': delta, 'placements': hits, 'reasons': reason,
            'mechanical_journal_candidate': not reason,
            'pair_provenance': 'observed_grid_delta_not_independent_tsumo_identity',
            'partial_raw_frames': [r['frame'] for r, d in zip(middle, raw_diffs)
                                   if d['kind'] == 'plus_one_partial'],
            'physical_identity_certified': False, 'writer_called': False}


def scan_cycles(rows: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    cycles, interruptions, stable, falling = [], [], None, []
    for row in rows:
        if stable is not None and row['epoch'][0] != stable['epoch'][0]:
            if falling:
                interruptions.append({'reason': 'reset', 'frame': row['frame']})
            stable, falling = None, []
        if row['state'] == 'STABLE':
            if stable is not None and falling:
                cycles.append(cycle_result(stable, falling, row))
            stable, falling = row, []
        elif row['state'] == 'TSUMO_FALL' and stable is not None:
            falling.append(row)
        else:
            if stable is not None:
                interruptions.append({'reason': row['state'], 'frame': row['frame'],
                                      'previous_stable': stable['frame']})
            stable, falling = None, []
    if falling:
        interruptions.append({'reason': 'unfinished_at_end', 'frame': falling[-1]['frame']})
    return cycles, interruptions


def matching_plateau(rows: list[dict[str, Any]], index: int, direction: int) -> dict[str, Any] | None:
    target = rows[index]
    while 0 <= index < len(rows):
        value = rows[index]
        if (value['state'] != 'STABLE' or value['grid'] != target['grid']
                or value['epoch'] != target['epoch'] or not value['control_ok']):
            return None
        if value['control_ok'] and value['machine_current_match']:
            return value
        index += direction
    return None


def availability_comparison(cycle: dict[str, Any], rows: list[dict[str, Any]],
                            indices: dict[int, int]) -> dict[str, Any]:
    start, end = cycle['start'], cycle['end']
    anchor = matching_plateau(rows, indices[start['frame']], -1)
    available = matching_plateau(rows, indices[end['frame']], 1)
    geometry = bool(cycle['placements'])
    reasons = [r for r in cycle['reasons'] if r not in
               ('baseline_not_machine_match', 'end_not_machine_match')]
    if anchor is None:
        reasons.append('no_prior_matching_same_grid_plateau')
    if available is None:
        reasons.append('no_later_matching_same_grid_same_action_plateau')
    before_counter = (start.get('accounting') or {}).get('after', {}).get('tsumo_count', {})
    after_counter = (end.get('accounting') or {}).get('after', {}).get('tsumo_count', {})
    delta = {str(c): after_counter.get(str(c), 0) - before_counter.get(str(c), 0) for c in COLORS}
    pair = cycle['delta']['pair'] or []
    return {'prior_same_grid_evidence': compact(anchor) if anchor else None,
            'available_same_grid_evidence': compact(available) if available else None,
            'observed_exit_frame': end['frame'],
            'observed_placement_interval': [start['frame'], end['frame']],
            'candidate': geometry and not reasons, 'reasons': reasons,
            'counter_delta_at_exit': delta,
            'counter_delta_matches_observed_pair': delta == {str(c): pair.count(c) for c in COLORS},
            'base_debt_known': False, 'physical_occurrence_known': False,
            'available_time_backdated': False}


def describe_side(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bases = [r for r in rows if r['state'] == 'STABLE' and r['control_ok']
             and r['machine_current_match']]
    per_reset = {}
    for row in bases:
        per_reset.setdefault(str(row['epoch'][0]), compact(row))
    cycles, interruptions = scan_cycles(rows)
    indices = {r['frame']: i for i, r in enumerate(rows)}
    for cycle in cycles:
        cycle['availability_comparison'] = availability_comparison(cycle, rows, indices)
    summary = Counter(c['delta']['kind'] for c in cycles)
    first = {kind: next(c for c in cycles if c['delta']['kind'] == kind)
             for kind in summary}
    candidates = [c for c in cycles if c['mechanical_journal_candidate']]
    same_color = [c for c in cycles if c['delta']['pair'] and len(set(c['delta']['pair'])) == 1]
    return {'state_counts': dict(Counter(r['state'] for r in rows)),
            'baseline_candidate_count': len(bases), 'first_by_reset': per_reset,
            'first_nonempty': next((compact(r) for r in bases if any(map(any, r['grid']))), None),
            'cycles': cycles, 'cycle_classification': dict(summary),
            'first_cycle_by_class': first, 'interruptions': interruptions,
            'candidate_count': len(candidates),
            'availability_candidate_count': sum(c['availability_comparison']['candidate'] for c in cycles),
            'same_color_cycle_frames': [c['end']['frame'] for c in same_color],
            'partial_raw_cycle_frames': [c['end']['frame'] for c in cycles if c['partial_raw_frames']]}


def scan_context(frames: dict[Any, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    sides, both_bases, seen = {side: [] for side in SIDES}, [], []
    with (LIVE / 'provisional_context.jsonl').open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            require(row['source_id'] == receipt['source_id'] and row['run_id'] == receipt['run_id'],
                    'context_source_run')
            frame = row['frame_idx']
            require(type(frame) is int and row['update']['call_index'] == line_number - 1,
                    'frame_call_index')
            require(not seen or frame == seen[-1] + STRIDE, 'clock_stride')
            require(row['available_frame'] == frame == row['update']['returned_frame_idx'], 'available')
            require(row['time_sec'] == row['update']['returned_time_sec'], 'return_time')
            seen.append(frame)
            for side in SIDES:
                value = summarize(row, side, frames)
                value['context_line'] = line_number
                sides[side].append(value)
            if all(sides[s][-1]['state'] == 'STABLE' and sides[s][-1]['control_ok']
                   and sides[s][-1]['machine_current_match'] for s in SIDES):
                both_bases.append(frame)
    require(seen == receipt['expected_frames'], 'expected_coverage')
    return {'contexts': len(seen), 'side_count': sum(map(len, sides.values())),
            'both_machine_baseline_frames': both_bases,
            'sides': {side: describe_side(values) for side, values in sides.items()}}


def verify_inputs() -> tuple[dict[str, str], dict[str, Any]]:
    require(sha(LIVE / 'COMPLETE') == COMPLETE_SHA, 'complete_sha')
    complete, plan = read(LIVE / 'COMPLETE'), read(LIVE / 'PLAN.json')
    require(type(complete['child_exit_code']) is int and complete['child_exit_code'] == 0, 'child')
    hashes = {str(LIVE / name): sha(LIVE / name) for name in LIVE_FILES}
    for name in LIVE_FILES[1:]:
        require(hashes[str(LIVE / name)] == complete['sha256'][name], 'input_sha:' + name)
    input_map = plan['input_and_code_sha256']
    wanted = ['src/board.py', 'src/puyo_core_bridge.py']
    for relative in wanted:
        path = SNAPSHOT / relative
        require(sha(path) == input_map[str(path)], 'snapshot_sha')
        hashes[str(path)] = sha(path)
    for path, digest in input_map.items():
        if '/site-packages/puyo_core/' in path:
            require(sha(Path(path)) == digest, 'native_sha')
            hashes[path] = digest
    for path in UNIT.glob('*.py'):
        hashes[str(path)] = sha(path)
    return hashes, read(LIVE / 'PROVISIONAL_CONTEXT_RECEIPT.json')


def main() -> int:
    output = UNIT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started, before = time.perf_counter(), {}
    try:
        before, receipt = verify_inputs()
        save(output / 'INPUTS.json', before)
        sys.path.insert(0, str(SNAPSHOT))
        core = importlib.import_module('src.puyo_core_bridge')
        require(Path(core.__file__).resolve() == SNAPSHOT / 'src/puyo_core_bridge.py', 'actual_core')
        import pytest
        with (output / 'PYTEST.log').open('x') as log:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(log), redirect_stderr(log):
                code = pytest.main([str(UNIT / 'test_probe.py'), '-q', '-p', 'no:cacheprovider'])
        require(code == 0, 'unit_test_failure')
        frames, schema = frame_index()
        result = scan_context(frames, receipt)
        save(output / 'FRAME_SCHEMA.json', schema)
        save(output / 'DIAGNOSIS.json', result)
        after = {path: sha(Path(path)) for path in before}
        require(before == after, 'input_changed')
        violations = [f'{p.name}:{n.name}' for p in UNIT.glob('*.py') for n in ast.walk(ast.parse(p.read_text()))
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > 50]
        require(not violations, 'function_over_50')
        save(output / 'RESULT.json', {'pid': os.getpid(), 'pytest_exit': int(code), 'seconds': time.perf_counter()-started,
             'before': before, 'after': after, 'function_over_50': violations,
             'actual_core': str(core.__file__), 'native_available': core.NATIVE_AVAILABLE,
             'quality_gate_clear': False, 'physical_identity_certified': False, 'writer_called': False,
             'scope': 'saved_mechanical_candidates_only_not_inventory_authorization'})
        artifacts = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        save(output / 'COMPLETE.json', {'sha256': artifacts, 'analysis_exit': 0})
        print(json.dumps({'pid': os.getpid(), 'exit': 0, 'seconds': time.perf_counter()-started}))
        return 0
    except BaseException as error:
        save(output / 'FAILED.json', {'pid': os.getpid(), 'error': repr(error), 'before': before})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
