"""原selected moduleの同一joinへ所有範囲限定で接続し、終了票を残す。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from typing import Any
import group_snapshot as G


def derived(parent: Any, side: Any) -> Any:
    class Context(parent):
        def perform(self, advisory: Any, frame: int, clock: float) -> None:
            assert 'tracker_snapshot_repair' not in self.state, 'snapshot_repeated_perform'
            evidence: dict[str, Any] = dict(frame=frame, G2=False, installed=False, restored=False)
            self.state['tracker_snapshot_repair'] = evidence
            self.stack.push(writer(self.state['output'], evidence))
            original = side.selected
            calls: list[Any] = []
            def selected() -> Any:
                assert not calls, 'snapshot_second_selected'
                module = original()
                join = module.parts().join
                calls.append(module)
                G.install(self.stack, join, G.group_type(self.pipe), evidence)
                assert module.parts().join is join, 'snapshot_split_join'
                return module
            with ExitStack() as scope:
                side.selected = selected
                def close(kind: Any, body: Any, trace: Any) -> bool:
                    if side.selected is selected:
                        side.selected = original
                        evidence['selected_restored'] = True
                    else:
                        evidence['selected_cleanup_error'] = 'foreign_selected'
                        if body is None:
                            raise RuntimeError('foreign_selected')
                    return False
                scope.push(close)
                return super().perform(advisory, frame, clock)
    return Context


def save(output: Any, state: Any) -> None:
    evidence = state['tracker_snapshot_repair']
    assert evidence['installed'] and evidence['restored'] and evidence['selected_restored'], 'snapshot_not_restored'
    assert not evidence.get('cleanup_error') and not evidence.get('selected_cleanup_error')
    saved = json.loads((output / 'TRACKER_SNAPSHOT_REPAIR.json').read_bytes())
    assert saved == evidence, 'snapshot_saved_mismatch'


def writer(output: Any, evidence: Any) -> Any:
    def close(kind: Any, body: Any, trace: Any) -> bool:
        evidence['body_error'] = None if body is None else repr(body)
        try:
            with (output / 'TRACKER_SNAPSHOT_REPAIR.json').open('x', encoding='utf-8') as stream:
                json.dump(evidence, stream, indent=2)
        except BaseException as error:
            evidence['save_error'] = repr(error)
            if body is None:
                raise
        return False
    return close
