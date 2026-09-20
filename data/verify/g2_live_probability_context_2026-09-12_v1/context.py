"""実Contextへ既存の確率経路を合成する。人工入力と検査専用driverは移植しない。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

KEY = 'live_probability_context'


@dataclass(frozen=True)
class Dependencies:
    """原factory生成後に読み込まれた同一型の既存部品を受け取る。"""

    modules: Callable[[], Any]
    quarantine: Any
    votes: Callable[[], Any]
    side: Any
    occurrence: Any
    baseline: Any
    snapshot: Any
    anchor: Any


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('live_probability_context:' + reason)


def core(parent: type, deps: Dependencies, end_frame: int) -> type:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            require(KEY not in self.state, 'duplicate_perform')
            require(type(frame) is int and 0 <= frame < end_frame, 'tracking_range')
            require('live_history_sink' in self.state, 'missing_original_sink')
            require(self.pipe._enable_match_transition_debounce is True, 'constructor_not_live_aligned')
            report = dict(frame=frame, end_frame=end_frame, stage='parts', error=None,
                          integer_permission=False, quality_gate_clear=False)
            self.state[KEY] = report
            try:
                parts = deps.modules()
                report['stage'] = 'original_perform'
                super().perform(advisory, frame, clock)
                report['stage'] = 'quarantine'
                deps.quarantine.install(self.stack, self.recovery, self.state)
                # 原Recovery自身の期限を使用し、候補側で延長しない。
                deadline = self.recovery_deadline()
                require(frame <= deadline <= end_frame, 'acquisition_after_tracking_end')
                report['stage'] = 'probabilistic_basis'
                observer = parts.actual.install(self.stack, self.recovery, self.state, frame, deadline)
                binding = parts.binding.install(self.stack, self.recovery, self.state, observer, end_frame)
                mode = parts.mode.install(self.stack, binding, self.state)
                parts.lease.install(self.stack, mode, self.state['repeat_scope_guard'].reset_lease)
                report['stage'] = 'qualified_prior_votes'
                deps.votes().install(self.stack, self.recovery, self.state, frame, deadline)
                report.update(stage='installed', acquisition_deadline=deadline)
                self.rows.append(dict(kind='probabilistic_context_installed', **report))
            except BaseException as error:
                report['error'] = repr(error)
                if self.error is None:
                    self.error = repr(error)
                raise

        def recovery_deadline(self) -> int:
            import sys
            deadline = sys.modules[type(self.recovery).__module__].I.DEADLINE
            require(type(deadline) is int, 'original_deadline_type')
            return deadline

    return Context


def compose(parent: type, deps: Dependencies, *, end_frame: int) -> type:
    require(type(end_frame) is int and end_frame > 0, 'tracking_end')
    selected = deps.side.derived(core(parent, deps, end_frame))
    retired = deps.occurrence.derived(selected)
    baseline = deps.baseline.derived(retired)
    snapshot = deps.snapshot.derived(baseline, deps.side)
    return deps.anchor.derived(snapshot)
