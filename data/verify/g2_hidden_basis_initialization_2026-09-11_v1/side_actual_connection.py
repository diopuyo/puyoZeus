"""固定片側resetを原Contextへ装着し、原実資格の前後と失敗段階を保存する。未実行候補。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import inflight_loader as L

ROOT = Path(__file__).resolve().parent.parent / 'g2_side_lease_connection_2026-09-11_v1'
PINS = {
    'side_lease.py': '08c7c3453b6c9e39e95948fe4126410649ce739560e87a37ab752201a4a7853b',
    'selected_side_lease.py': '19b650d434e7d872d0a5c10c98bbb57aa1043b904c27b61f220c4241c191a3c8',
}


def selected() -> Any:
    for name, digest in PINS.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, 'side_actual_source'
    old = L.load('_g2_actual_side_lease_base', ROOT / 'side_lease.py')
    return L.load('_g2_actual_side_lease_selected', ROOT / 'selected_side_lease.py', {'side_lease': old})


def snapshot(context: Any, module: Any) -> dict[str, Any]:
    journal, pipe = context.factory.provider.journal, context.pipe
    runtime = journal.controller.instances[id(pipe)]
    join = module.parts().join
    sides = {}
    for side in ('1P', '2P'):
        history = runtime.histories[side]
        assert journal.epoch(pipe, side) == history.epoch, 'actual_native_epoch'
        sides[side] = dict(epoch=history.epoch, history=asdict(history), history_id=id(history),
                           generation=asdict(journal.tracker.generation(side)))
    return dict(sides=sides, pipe_id=id(pipe), history_dict_id=id(runtime.histories),
        true_clocks={name: join._vrepr(getattr(pipe, name)) for name in join.TRUE_CLOCK_FIELDS},
        shared={name: join._vrepr(getattr(pipe, name)) for name in join.SHARED_FIELDS})


def verify(before: Any, after: Any, lease: Any) -> None:
    assert after['sides']['2P'] == before['sides']['2P'], 'actual_other_side_changed'
    assert after['sides']['1P']['epoch'] == before['sides']['1P']['epoch'] + 1
    assert after['sides']['1P']['generation']['reset_epoch'] == before['sides']['1P']['generation']['reset_epoch'] + 1
    for key in ('pipe_id', 'history_dict_id', 'true_clocks', 'shared'):
        assert after[key] == before[key], 'actual_shared_changed:' + key
    assert lease.outer_calls == lease.native_calls == lease.depth == 0, 'actual_full_reset_called'


def save(context: Any, report: Any, original_error: Any) -> None:
    recovery = context.recovery
    lease = context.state['repeat_scope_guard'].reset_lease
    report.update(lease_events=lease.events, recovery_rows=None if recovery is None else recovery.rows[-2:],
                  complete_recovery_evidence='LIVE_EMPTY_RESET.json', source_sha=PINS,
                  quality_gate_clear=False, actual_video=False)
    try:
        with (context.state['output'] / 'SIDE_ACTUAL_CONNECTION.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    except BaseException as error:
        if original_error is None:
            raise
        context.rows.append(dict(stage='side_failure_save_failed', error=repr(error)))


def run(context: Any, original: Any, advisory: Any, frame: int, clock: float) -> None:
    report, error, module = dict(frame=frame, clock=clock, stage='install'), None, None
    try:
        module = selected()
        lease = context.state['repeat_scope_guard'].reset_lease
        module.install(context.stack, lease)
        report['stage'] = 'before_snapshot'
        report['before'] = snapshot(context, module)
        report['stage'] = 'original_context_perform'
        original(advisory, frame, clock)
        report['after'] = snapshot(context, module)
        verify(report['before'], report['after'], lease)
        report.update(stage='side_reset_connected', full_reset_calls=0, other_side_unchanged=True)
    except BaseException as caught:
        error = caught
        kind = getattr(sys.modules.get('_g2_selected_side_join'), 'SideJoinRejected', ())
        rejected = isinstance(caught, kind)
        stage = 'side_install_failed' if report['stage'] == 'install' else 'side_operation_failed'
        report.update(failed_stage=report['stage'], stage='side_preflight_rejected' if rejected else stage,
                      error=repr(caught), reason=getattr(caught, 'reason', None), details=getattr(caught, 'report', None))
        context.error = repr(caught)
        raise
    finally:
        save(context, report, error)


def derived(parent: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            run(self, super().perform, advisory, frame, clock)
    return Context
