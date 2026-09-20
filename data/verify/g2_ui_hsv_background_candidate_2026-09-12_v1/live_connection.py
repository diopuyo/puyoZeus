"""原Readerの同call HSVへ私有UI候補を接続。元HSV例外処理と所有解除を維持する。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any
import ui_background_policy as P

SOURCE = Path(__file__).resolve().parents[3] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/image_reader.py'


class DiagnosticStop(BaseException):
    """原stepのexcept Exceptionで診断欠測が黙殺されることを防ぐ。私有検証専用。"""


def witness(recovery: Any, caller: Any, reader: Any, image: Any, region: Any, side: str) -> dict | None:
    if caller.f_code not in recovery.journal.codes or caller.f_locals.get('side') != side:
        return None  # 別side/元step以外の既存利用は一切変えない。
    local, item = caller.f_locals, recovery.journal.active
    P.require(item is not None and item['frame'] is caller and item['pipe'] is recovery.pipe, 'original_J_call')
    P.require(local['self'] is recovery.pipe and recovery.pipe._reader is reader, 'reader_pipe')
    P.require(local['frame_bgr'] is image and local['region_for_hsv'] is region, 'same_image_region')
    P.require(local['frame_idx'] == item['scope']['frame_idx']
              and local['time_sec'] == item['scope']['time_sec']
              and item['scope']['side'] == side, 'same_clock_side')
    return dict(scope=dict(item['scope']), call_token=item['token'],
                stable=local['sm'].context.state.value == 'stable')


class Connection:
    def __init__(self, recovery: Any, matcher: Any, output: Path, side: str) -> None:
        self.recovery, self.matcher, self.side = recovery, matcher, side
        self.reader = recovery.pipe._reader
        P.require('read_board_hsv_only' not in vars(self.reader), 'foreign_reader_hook')
        self.original = self.reader.read_board_hsv_only
        P.require(Path(self.original.__func__.__code__.co_filename).resolve() == SOURCE, 'original_reader_code')
        self.output, self.rows, self.error = output, 0, None
        self.snapshot = recovery.journal.snapshot.__func__.__globals__['board']
        self.wrapper = MethodType(self.forward, self.reader)
        self.stream = (output / 'UI_HSV_EVIDENCE.jsonl').open('x', encoding='utf-8')
        try:
            self.reader.read_board_hsv_only = self.wrapper
        except BaseException:
            self.stream.close()
            raise

    def forward(self, reader: Any, image: Any, region: Any) -> Any:
        caller = sys._getframe(1)
        try:
            P.require(reader is self.reader and reader.read_board_hsv_only is self.wrapper, 'hook_owner')
            call = witness(self.recovery, caller, reader, image, region, self.side)
        except Exception as error:
            self.error = repr(error)
            raise DiagnosticStop(self.error) from error
        original = self.original(image, region)  # 元例外は元stepと同じ扱い。
        if call is None:
            return original
        try:
            before = self.snapshot(original)
            result, proof = P.normalize(original, image, region, self.matcher, stable=call['stable'])
            P.require(self.snapshot(original) == before, 'original_HSV_mutated')
            row = call | dict(original_HSV=before, normalized_HSV=self.snapshot(result), **proof)
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.stream.flush()
            self.rows += 1
            return result
        except Exception as error:
            self.error = repr(error)
            raise DiagnosticStop(self.error) from error

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        owned = vars(self.reader).get('read_board_hsv_only') is self.wrapper
        if owned:
            del self.reader.read_board_hsv_only
        try:
            self.stream.close()
            restored = owned and self.reader.read_board_hsv_only == self.original
            with (self.output / 'UI_HSV_STATUS.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(rows=self.rows, restored=restored, closed=self.stream.closed,
                               error=self.error, quality_gate_clear=False), stream, indent=2)
            P.require(restored, 'restore_owner')
        except Exception as error:
            self.error = 'close:' + repr(error)
            try:
                print('UI_HSV_CLOSE_ERROR=' + self.error, file=sys.stderr, flush=True)
            except Exception:
                pass
            if body is None:
                raise DiagnosticStop(self.error) from error
        return False


def install(stack: Any, recovery: Any, matcher: Any, output: Path, side: str = '1P') -> Connection:
    value = Connection(recovery, matcher, output, side)
    stack.push(value.close)
    return value
