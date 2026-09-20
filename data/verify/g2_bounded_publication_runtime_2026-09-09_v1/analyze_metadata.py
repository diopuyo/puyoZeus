"""新実走の採録全非ラベル列を元appendで再生し、観測非干渉を保存比較する。"""
from __future__ import annotations
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import live_cli as L
O = L.M.O

LIVE = L.ROOT.parent / 'video38_bounded_pending_publication_live_2026-09-09_v1'
OLD = L.ROOT.parent / 'video38_floating_publication_live_2026-09-09_v1'
STREAMS = ('frames.jsonl', 'current_entry.jsonl', 'resolved_grace_guard.jsonl', 'chigiri_completion.jsonl',
    'provisional_context.jsonl', 'consumer_publication.jsonl')


def decode(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if 'nonfinite_float' in value:
        return float(value['nonfinite_float'])
    if 'mapping' in value:
        return {decode(k): decode(v) for k, v in value['mapping']}
    if 'sequence_type' in value:
        values = [decode(v) for v in value['items']]
        return tuple(values) if value['sequence_type'] == 'tuple' else values
    if 'ndarray_dtype' in value:
        import numpy as np
        result = np.array(decode(value['values']), dtype=value['ndarray_dtype'])
        L.F.require(list(result.shape) == value['shape'], 'metadata_decode_shape')
        return result
    raise ValueError('metadata_decode_unsupported')


def collector() -> Any:
    path = L.ROOT.parents[2] / 'scripts/diagnose_video38_confirmed_collapse_v1.py'
    base = L.load('_metadata_replay_loader', path)
    result = base.load_collector()
    O.validate_collector(result)
    return result


def replay(c: Any, row: Any) -> dict[str, Any]:
    acc = c._LeanNpzAccumulator()
    args = decode(row['arguments'])
    acc.append(**args)
    actual = {name: O.serial(getattr(acc, name)[0]) for name in O.STORED_KEYS}
    differences = [name for name in sorted(O.STORED_KEYS) if actual[name] != row['stored_nonlabel_row'][name]]
    return {'frame': args['frame_idx'], 'side': args['side'], 'differences': differences,
        'game': actual['game_idxs'], 'image_tag': actual['stable_persistence_confidences'],
        'match_end_locked': actual['match_end_lockeds'], 'lockdown': actual['post_match_lockdown_actives'],
        'board_provenance': actual['board_provenances']}


def verify_inputs() -> dict[str, str]:
    L.M.verify(LIVE)
    L.P.verify(LIVE)
    complete = L.F.P.read(LIVE / 'COMPLETE')
    child = L.F.P.read(LIVE / 'CHILD_EXIT.json')
    L.F.require(complete['child_exit_code'] == child['child_exit_code'] == 0, 'metadata_analysis_child_failed')
    L.F.require((L.M.REQUIRED | L.P.REQUIRED) <= complete['sha256'].keys(), 'metadata_analysis_complete_missing')
    bound = {str(LIVE / n): h for n, h in complete['sha256'].items()}
    L.F.require(all(L.F.sha(Path(p)) == h for p, h in bound.items()), 'metadata_analysis_artifact_changed')
    bound |= {str(LIVE / 'COMPLETE'): L.F.sha(LIVE / 'COMPLETE')}
    plan = L.F.P.read(LIVE / 'PLAN.json')['input_and_code_sha256']
    L.F.require(all(L.F.sha(Path(p)) == h for p, h in plan.items()), 'metadata_analysis_plan_changed')
    return bound | plan | {str(OLD / n): L.F.sha(OLD / n) for n in STREAMS}



def publication_delta(x: Any, accepted: list[tuple[int, str]]) -> dict[str, Any]:
    changes, counts = [], Counter()
    allowed = {'confirmed', 'probability', 'board_none_reason'}
    windows = {'C1': ('1P', 34038, 34144), 'C2': ('2P', 34038, 34290),
        'N1': ('1P', 33726, 33756), 'N2': ('2P', 33726, 33756)}
    sources = (x.rows(OLD / 'consumer_publication.jsonl'), x.rows(LIVE / 'consumer_publication.jsonl'))
    for old, new in zip(*sources, strict=True):
        frame = new['frame_idx']
        L.F.require(old['frame_idx'] == frame and old['time_sec'] == new['time_sec'], 'publication_delta_clock')
        for side in ('1P', '2P'):
            a, b = old['returned'][side], new['returned'][side]
            changed = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
            if changed:
                L.F.require(set(changed) <= allowed and a['confirmed'] is None
                    and a['board_none_reason'] == L.P.PENDING and b['confirmed'] is not None
                    and b['board_none_reason'] is None, 'publication_unexpected_change')
                changes.append((frame, side))
            for label, (target, first, last) in windows.items():
                if side == target and first <= frame <= last:
                    counts[label + ':side_rows'] += 1
                    counts[label + ':changed'] += bool(changed)
    L.F.require(sorted(changes) == accepted, 'pending_status_publication_join')
    return {'changed_frame_sides': changes, 'controls': dict(counts),
        'only_fresh_pending_current_changed': True, 'controls_visually_certified': False}


def publication_analysis(output: Path) -> dict[str, Any]:
    path = L.C.A.ROOT / 'analyze_live.py'
    x = L.load('_bounded_publication_analysis', path)
    x.OLD = OLD
    x.SELECTED = (34702, 34704, 35784, 35806, 35808, 35810, 35838, 35846, 35850, 36002, 36298)
    report = x.compare_context(LIVE)
    L.F.require(not report['upstream_differences'], 'upstream_context_changed')
    selected = [r for r in report['selected_frames'] if r['side'] == '2P' and r['frame'] in (35806, 35808)]
    L.F.require(len(selected) == 2 and all(r['before']['board_sha256'] is None
        and r['after']['board_sha256'] == 'ae72c245504451f141c8d4dd13cd05b93edc5ce06cb3354c6f4b78ff3e5649f8'
        for r in selected), 'pending_real_nine_not_restored')
    stats = L.F.P.read(LIVE / L.P.STATUS)
    accepted = sorted({(r['frame'], r['side']) for r in stats['rows'] if r['accepted']})
    report['same_background_publication_delta'] = publication_delta(x, accepted)
    report['pending_unique_accepted'] = accepted
    report['ticket_calls_not_publication_count'] = len(stats['rows'])
    report['quality_gate_clear'] = False
    L.F.P.write(output / 'PUBLICATION_ANALYSIS.json', report)
    return {'upstream_differences': 0, 'counts': report['counts'], 'pending_unique_accepted': accepted}


def execute(output: Path) -> dict[str, Any]:
    before = verify_inputs() | {str(p): L.F.sha(p) for p in (Path(__file__), L.C.A.ROOT / 'analyze_live.py')}
    c, results = collector(), []
    with (LIVE / O.ENTRIES).open() as stream:
        for line in stream:
            for append in json.loads(line)['appends']:
                results.append(replay(c, append))
    publication = publication_analysis(output)
    same = {name: L.F.sha(OLD / name) == L.F.sha(LIVE / name) for name in STREAMS}
    differences = [row for row in results if row['differences']]
    L.F.require(not differences, 'metadata_append_replay_differences')
    report = {'source_run': str(LIVE), 'same_background_stream_sha_equal': same, 'append_count': len(results),
        'all_nonlabel_columns': sorted(O.STORED_KEYS), 'replay_differences': differences, 'appends': results,
        'publication_context': publication, 'image_tags': dict(Counter(str(r['image_tag']) for r in results)),
        'games': dict(Counter(str(r['game']) for r in results)), 'ground_truth_evaluated': False,
        'labels_read': False, 'quality_gate_clear': False, 'actual_collect_reexecuted_in_analysis': False}
    L.F.P.write(output / 'METADATA_ANALYSIS.json', report)
    after = {p: L.F.sha(Path(p)) for p in before}
    L.F.require(before == after, 'metadata_analysis_input_changed')
    return {'before': before, 'after': after, 'append_count': len(results), 'replay_differences': len(differences),
        'same_background_stream_sha_equal': same, 'quality_gate_clear': False}


def main() -> int:
    output = L.ROOT / sys.argv[1]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = execute(output)
    result.update(pid=os.getpid(), actual_exit=0, seconds=time.perf_counter() - started)
    L.F.P.write(output / 'RESULT.json', result)
    L.F.P.write(output / 'ANALYSIS_COMPLETE.json', {'sha256': {n: L.F.sha(output / n)
        for n in ('RESULT.json', 'METADATA_ANALYSIS.json', 'PUBLICATION_ANALYSIS.json')}})
    print({k: v for k, v in result.items() if k not in ('before', 'after')}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
