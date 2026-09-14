"""原追跡observe直前の同call局所値を保存する。入力・状態・例外を変更しない。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Callable

FIRST, LAST = 35294, 35338


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('tracking_observation:' + reason)


def capture(mode: Any, item: dict, result: Any, local_capture: Callable) -> dict:
    recovery, frame = mode.connection.recovery, item['frame']
    require(frame.f_code in recovery.journal.codes, 'original_step_code')
    require(recovery.journal.active is None, 'journal_not_complete')
    require(item['pipe'] is recovery.pipe, 'original_pipe')
    require(item['scope']['side'] == mode.connection.binding.scope[-1], 'side')
    require(frame.f_locals['frame_idx'] == item['scope']['frame_idx'], 'frame')
    snapshot = recovery.journal.snapshot.__func__.__globals__['board']
    typed = recovery.state['postcommit_current_receiver'].rec.side_value(result)
    row = local_capture(frame.f_locals, result, typed, snapshot)
    context = frame.f_locals['sm'].context
    counters = [[int(r), int(c), int(v)] for (r, c), v in
                sorted(context.stable_recovery_counters.items())]
    return row | dict(scope=dict(item['scope']), call_token=item['token'],
                      stable_recovery_counters=counters, observation_before_tracking=True)


class Connection:
    def __init__(self, mode: Any, output: Path, local_capture: Callable) -> None:
        require('observe' not in vars(mode), 'foreign_observe')
        self.mode, self.original, self.local_capture = mode, mode.observe, local_capture
        self.stream = (output / 'TRACKING_LOCAL_OBSERVATIONS.jsonl').open('x', encoding='utf-8')
        self.path, self.rows, self.seen = output, 0, set()
        self.error: str | None = None
        self.wrapper = MethodType(self.forward, mode)
        mode.observe = self.wrapper

    def forward(self, mode: Any, item: dict, result: Any, error: Any) -> Any:
        failure = None
        try:
            require(mode is self.mode and mode.observe is self.wrapper, 'owner')
            frame = item['scope']['frame_idx']
            if FIRST <= frame <= LAST:
                require(item['token'] not in self.seen, 'duplicate_call')
                require(self.rows < LAST - FIRST + 1, 'bounds')
                require(error is None, 'unexpected_step_error')
                row = capture(mode, item, result, self.local_capture)
                self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                self.stream.flush()
                self.seen.add(item['token'])
                self.rows += 1
        except BaseException as caught:
            self.error, failure = repr(caught), caught
        returned = self.original(item, result, error)  # 元例外を優先し、一度だけ実行する。
        if failure is not None:
            raise failure
        return returned

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        failure = None
        owned = vars(self.mode).get('observe') is self.wrapper
        if owned:
            del self.mode.observe
        restored = owned and self.mode.observe == self.original
        try:
            self.stream.close()
            with (self.path / 'TRACKING_LOCAL_STATUS.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(rows=self.rows, bounds=[FIRST, LAST], restored=restored,
                               closed=self.stream.closed, error=self.error,
                               quality_gate_clear=False), stream, indent=2)
            require(restored, 'restore')
        except BaseException as caught:
            failure = caught
            self.error = 'close:' + repr(caught)
            try:
                print('TRACKING_LOCAL_CLOSE_ERROR=' + self.error, file=sys.stderr, flush=True)
            except BaseException:
                pass  # 診断出力失敗で元例外を置換しない。STATUS欠測は検収失敗。
        if body is None and failure is not None:
            raise failure
        return False


def install(stack: Any, mode: Any, output: Path, local_capture: Callable) -> Connection:
    value = Connection(mode, output, local_capture)
    stack.push(value.close)
    return value
