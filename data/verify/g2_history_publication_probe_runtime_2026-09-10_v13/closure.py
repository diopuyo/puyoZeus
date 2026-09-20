"""診断の計算閉鎖と履歴達成を分離する。旧失敗runに書き込まない。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import common as K
import recording as R
import goals as G
import finalizer_connection as FINAL

ENGINE = 'PUBLICATION_ENGINE.json'
PERMISSIONS = dict(current_permission=False, quality_gate_clear=False,
    production_permission=False, physical_certified=False, whole_writer_noninterference_certified=False)
STATUS_NAMES = ('CHIGIRI_COMPLETION_STATUS.json', 'COLLECTOR_METADATA_STATUS.json',
    'FLOATING_EXIT_STATUS.json', 'HIDDEN_PROBABILITY_STATUS.json', 'HSV_WITNESS_STATUS.json',
    'NORMAL_COMPLETION_STATUS.json', 'PENDING_FRESH_CURRENT_STATUS.json',
    'PRIVATE_CONSUMER_STATUS.json', 'PROVISIONAL_CONTEXT_STATUS.json', 'RESOLVED_GRACE_STATUS.json',
    'SPLIT_EVIDENCE_STATUS.json', 'ATOMIC_JOURNAL_STATUS.json', R.STATUS, G.PUB_STATUS, G.OUTER_STATUS)


def saved_closures(output: Path) -> dict[str, Any]:
    result = {}
    for name in STATUS_NAMES:
        value = K.read(output / name)
        K.require(value.get('closed') is True, 'saved_sink_not_closed:' + name)
        for key in ('errors', 'failures', 'error', 'sticky_error', 'guard_error'):
            if key in value:
                K.require(value[key] is None or value[key] == [], 'saved_sink_error:' + name + ':' + key)
        for key in ('installed', 'guard_installed'):
            K.require(key not in value or value[key] is True, 'saved_not_installed:' + name)
        K.require('active' not in value or value['active'] is False, 'saved_active:' + name)
        result[name] = {k: v for k, v in value.items() if k in
            ('closed', 'errors', 'failures', 'sticky_error', 'guard_error', 'active', 'installed', 'guard_installed')}
    return result


def finish(env: Any, state: Any, rec: Any, sink: Any, handles: Any, binding: Any) -> dict[str, Any]:
    K.require(rec.frames == len(K.FRAMES) and rec.frame == K.FRAMES[-1] and sink.updates == list(K.FRAMES),
              'actual_collection_coverage')
    K.require(rec.model_loads and rec.pipeline_receipt.get('board_cnn_device') == 'cuda:0', 'actual_cuda_model_missing')
    adopted = K.read(state['output'] / 'BASELINE_ADOPTION_STATUS.json')
    K.require(adopted['closed'] and adopted['error'] is None, 'baseline_adoption_unclosed')
    raw = state['raw_input_bridge']
    K.require(raw.closed and raw.error is None and raw.active is None and raw.pending is None
              and raw.last_frame == K.FRAMES[-1], 'raw_input_not_closed')
    control = env['factory'].controller
    K.require(control.sticky_error is None and not control.calls and not control.tickets, 'history_controller_unclosed')
    K.require(env['factory'].provider.link.error is None and binding['provider'].error is None
        and binding['adapter'].controller.active is None, 'directional_or_handoff_failure')
    K.require(sink.closed and not sink.errors and handles and all(v.closed for v in handles.values()), 'sink_stream_unclosed')
    observers = R.observer_status(state)
    env['addon'].journal.finish(state)
    saved_status = saved_closures(state['output'])
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    goal = FINAL.evaluate(G, rows, control.legal, state['output'], state)
    K.write(state['output'] / 'PUBLICATION_GOAL.json', goal)
    K.write(state['output'] / 'DIRECTIONAL_PROVIDER.json', binding['provider'].records)
    K.write(state['output'] / 'HISTORY_DERIVATION.json', env['addon'].derivation)
    return dict(goal=goal, observers=observers, saved_status=saved_status, baseline_adoption=adopted,
        streams_closed={p: v.closed for p, v in handles.items()},
        frame_count=rec.frames, last_frame=rec.frame, row_count=rec.rows,
        snapshots=rec.snapshots, pipeline=rec.pipeline_receipt, model_loads=rec.model_loads,
        journal=dict(steps=state['atomic_journal_observer'].steps, enqueues=state['atomic_journal_observer'].enqueues),
        historical_controller_class=type(control).__name__, **PERMISSIONS)


def seal(output: Path, summary: Any, receipt: Any) -> None:
    K.require(summary['references_restored'] is True and summary['guards_unchanged'] is True, 'unrestored_or_changed')
    names = sorted(p.name for p in output.iterdir() if p.is_file())
    K.require('COMPLETE' not in names and 'CHILD_EXIT.json' not in names and ENGINE not in names, 'exclusive_engine')
    K.write(output / ENGINE, dict(schema='history-publication-probe-engine/v1', computation_closed=True,
        bounds=K.bounds(), summary=summary, guard_sha256=receipt['input_and_code_sha256'],
        sha256={n: K.sha(output / n) for n in names}, **PERMISSIONS))


def verify_engine(output: Path, engine: Any, resources: list[Any]) -> None:
    K.require(engine['computation_closed'] is True and engine['bounds'] == K.bounds(), 'engine_bounds')
    K.require(all(engine.get(k) is False for k in PERMISSIONS), 'engine_permission')
    K.require(engine['summary']['references_restored'] is True and engine['summary']['guards_unchanged'] is True,
              'engine_restore')
    K.require(all(K.sha(output / n) == h for n, h in engine['sha256'].items()), 'engine_artifact_changed')
    K.require(all(K.sha(Path(p)) == h for p, h in engine['guard_sha256'].items()), 'finalizer_guard_changed')
    entry = K.read(output / 'ENTRY_RESULT.json')
    K.require(type(entry['pid']) is int and entry['exit_code'] == 0 and type(entry['exit_code']) is int,
              'actual_entry_receipt')
    K.require(all(r['pid'] == entry['pid'] for r in resources), 'resource_child_pid')


def finalize(output: Path, code: int, resource_code: int, resources: Path) -> dict[str, Any]:
    K.require(type(code) is int and type(resource_code) is int and 0 <= code <= 255 and 0 <= resource_code <= 255,
              'actual_exit_codes_required')
    K.require(not (output / 'COMPLETE').exists() and not (output / 'CHILD_EXIT.json').exists(), 'exclusive_finalize')
    K.write(output / 'CHILD_EXIT.json', dict(child_exit_code=code, resource_guard_exit=resource_code, source='actual_wait'))
    if code != 0 or resource_code != 0:
        return dict(status='child_or_resource_failed', child_exit_code=code, **PERMISSIONS)
    K.require(resources.resolve() == Path(str(output.resolve()) + '.resources.jsonl'), 'resource_path')
    rows = [json.loads(line) for line in resources.read_text().splitlines()]
    K.require(rows and all(r['safety_stop'] is False for r in rows), 'resource_guard_unobserved_or_stopped')
    engine = K.read(output / ENGINE)
    verify_engine(output, engine, rows)
    result = dict(status='diagnostic_computation_closed', computation_closed=True,
        current_event_observed=engine['summary']['goal']['current_event_observed'],
        outer_publication_observed=engine['summary']['goal']['outer_publication_observed'],
        engine_sha256=K.sha(output / ENGINE), child_receipt_sha256=K.sha(output / 'CHILD_EXIT.json'),
        resource_sha256=K.sha(resources), **PERMISSIONS)
    K.write(output / 'COMPLETE', result)
    return result
