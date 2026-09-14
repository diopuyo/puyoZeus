"""原80更新・原reset資格から確率基準/次手へ進む人工入力CPU対象試験。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any
import proof_cpu_connected as DRIVER
import tracking_loader as L
import target_inputs as INPUT
import sm_trace

ORIGINAL_DERIVED = DRIVER.P.derived
TRACKING_END_FRAME = 36298


class Context(DRIVER.Context):
    def perform(self, advisory: Any, frame: int, clock: float) -> None:
        # 実v12のactual_collector_kwargsはdebounce=True、旧CPU constructorはFalse。
        # prefix不変のままreset境界だけ設定を揃える限定診断。全設定同値の主張ではない。
        parts = L.modules()
        old_debounce = self.pipe._enable_match_transition_debounce
        parts.mode.M.patch(self.stack, self.pipe, '_enable_match_transition_debounce', True)
        with (self.state['output'] / 'TARGET_CONFIG_ALIGNMENT.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(flag='enable_match_transition_debounce', old=old_debounce, new=True,
                applied_before_reset=True, entire_constructor_equivalent=False,
                evidence='live_v12/PLAN.json:actual_collector_kwargs', quality_gate_clear=False), stream)
        super().perform(advisory, frame, clock)
        deadline = sys.modules[type(self.recovery).__module__].I.DEADLINE
        observer = parts.actual.install(self.stack, self.recovery, self.state, frame, deadline)
        binding = parts.binding.install(self.stack, self.recovery, self.state, observer, TRACKING_END_FRAME)
        mode = parts.mode.install(self.stack, binding, self.state)
        parts.lease.install(self.stack, mode, self.state['repeat_scope_guard'].reset_lease)
        sm_trace.install(self.stack, self.pipe, self.state['output'])


def verify(recovery: Any, lease: Any, factory: Any, state: Any, trace: Any, prior: Any) -> dict[str, Any]:
    mode = state['probabilistic_tracking_mode']
    assert mode.activation is not None and mode.error is None and mode.native is not None
    L.modules().lease.verify(mode, lease)
    lease.archive.verify()
    old, saved, record = recovery.archive[0]
    assert old.owner.state is saved and asdict(saved) == record['owner']
    assert recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None
    assert factory.controller.history.get('1P') is None and not mode.native.pending
    if INPUT.BASIS_ONLY:
        assert not mode.applied and not mode.native.seen_occurrences
    else:
        assert mode.applied and mode.native.seen_occurrences
    assert factory.provider.journal.steps == 210 and len(trace) == 25
    receiver = state['postcommit_current_receiver']
    assert not receiver.errors and receiver.released == prior['released']
    result = dict(target_path_completed=not INPUT.BASIS_ONLY, basis_only=INPUT.BASIS_ONLY,
        actual_J_steps=factory.provider.journal.steps,
        probabilistic_basis_frame=mode.activation['frame'], physical_transitions=len(mode.applied),
        native_consumptions=len(mode.native.seen_occurrences), old_archive_unchanged=True,
        artificial_inputs=True, actual_video_run=False, integer_continuation_verified=False,
        full_finalizer_verified=False, quality_gate_clear=False)
    with (state['output'] / 'PROBABILISTIC_TARGET_RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def derived() -> Any:
    previous = ORIGINAL_DERIVED()
    values = dict(previous.__globals__)
    inputs, boundary = values['I'], values['B']
    def shifted(original: Any) -> Any:
        module = inputs.shifted(original)
        old_install = module.install
        def install(*args: Any) -> Any:
            return INPUT.install(old_install, inputs.RESET, *args)
        return N(**(vars(module) | dict(install=install)))
    def frame_state(factory: Any, lease: Any, frame: int) -> Any:
        mode = lease.recovery.state.get('probabilistic_tracking_mode')
        if mode is None or mode.native is None:
            return boundary.frame_state(factory, lease, frame)
        L.modules().lease.verify(mode, lease)
        value = mode.connection.registry.current(mode.connection.binding)
        return dict(frame=frame, current=False, owner=False, probabilistic_owner=True,
                    probability_frame=value.frame, old_evidence_leaked=False)
    def retire(lease: Any, factory: Any) -> None:
        lease.archive.verify()  # 元の「新整数owner検収」はこの別modeの合格に流用しない。
    values.update(I=N(**(vars(inputs) | dict(shifted=shifted))),
                  B=N(frame_state=frame_state, retire=retire), verify=verify)
    generated = FunctionType(previous.__code__, values)
    def extend(context: Any, result: Any) -> None:
        generated(context, result)
        result.update(original_prefix_collector_J_S=True, original_collector_J_S=False,
                      target_probabilistic_mode=True, quality_gate_clear=False)
    return extend


TARGET = N(derived=derived)


def dispatch() -> Any:
    return TARGET.derived()


def main() -> int:
    roots = [L.ROOT, L.ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1',
             L.ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1',
             L.ROOT.parent / 'g2_belief_hidden_landing_2026-09-11_v1']
    paths = [path for root in roots for path in root.glob('*.py')]
    sha = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = sha()
    DRIVER.C, DRIVER.Context = L.CONNECTION, Context
    DRIVER.P.TARGET, DRIVER.P.derived = TARGET, dispatch
    code = DRIVER.main()
    unchanged = before == sha()
    output = DRIVER.P.R.Q.ROOT / sys.argv[1]
    with (output / 'PROBABILISTIC_TARGET_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code,
                       quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
