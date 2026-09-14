"""metadataを逐次保存し、採録関数が読まないgray画像はdescriptorに限定する。"""
from __future__ import annotations
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'g2_collector_metadata_capture_2026-09-09_v1'
spec = importlib.util.spec_from_file_location('_bounded_metadata_original', PRIOR / 'observer.py')
O = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = O
spec.loader.exec_module(O)
OWN = ('bounded.py', 'test_bounded.py', 'run_cpu.py', 'CONTRACT.md')
GRAY = 'motion_prev_gray'
KEY, REQUIRED = O.KEY, O.REQUIRED


def guards() -> dict[str, str]:
    return O.guards() | {str(ROOT / n): O.sha(ROOT / n) for n in OWN}


def image_descriptor(value: Any) -> Any:
    if value is None:
        return None
    import hashlib
    import numpy as np
    O.require(type(value) is np.ndarray and value.dtype == np.uint8 and value.ndim == 2, 'gray_shape_type')
    return {'excluded_image_payload': True, 'reason': 'not_read_by_process_side_lean',
        'shape': list(value.shape), 'dtype': str(value.dtype),
        'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}


class RowStream:
    def __init__(self, output: Path) -> None:
        self.stream = (output / O.ENTRIES).open('x')
        self.count, self.append_count = 0, 0

    def append(self, row: dict[str, Any]) -> None:
        frame = O.FIRST + self.count // len(O.SIDES) * O.STRIDE
        side = O.SIDES[self.count % len(O.SIDES)]
        O.require(frame <= O.LAST, 'bounded_excess_rows')
        self.append_count = O.validate_row(row, frame, side, self.append_count)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        self.count += 1


class Sink(O.Sink):
    def __init__(self, collector: Any, output: Path) -> None:
        super().__init__(collector, output)
        self.before = guards()
        self.rows = RowStream(output)

    def close(self) -> None:
        O.require(not self.closed and not self.busy, 'bounded_close_state')
        self.rows.stream.close()
        self.closed = True
        O.write(self.output / O.STATUS, {'closed': True, 'errors': self.errors, 'rows': self.rows.count,
            'label_columns_read': False, 'quality_gate_clear': False, 'streaming': True,
            'image_payload_retained': False})


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any], *, enabled: bool = False) -> None:
    if not enabled:
        return
    O.require(type(enabled) is bool and KEY not in state, 'bounded_install_reentry')
    O.validate_collector(collector)
    sink, original = Sink(collector, state['output']), collector._process_side_lean
    previous_serial = O.serial
    def serial(value: Any) -> Any:
        if type(value) is collector._SideState:
            return {'dataclass_type': type(value).__name__, 'fields': {f.name:
                image_descriptor(getattr(value, f.name)) if f.name == GRAY
                else previous_serial(getattr(value, f.name)) for f in fields(value)}}
        return previous_serial(value)
    stack.callback(sink.close)
    stack.callback(setattr, O, 'serial', previous_serial)
    stack.callback(setattr, collector, '_process_side_lean', original)
    O.serial = serial
    collector._process_side_lean = sink.wrapper(original)
    state[KEY] = sink


def finish(state: dict[str, Any]) -> None:
    sink = state[KEY]
    O.require(sink.closed and not sink.errors and sink.before == guards(), 'bounded_failed_or_changed')
    counts = O.saved_counts(sink.output)
    O.require(counts == (sink.rows.count, sink.rows.append_count), 'bounded_saved_counts')
    O.write(sink.output / O.RECEIPT, {'guards': sink.before, 'rows': counts[0], 'actual_append_count': counts[1],
        'sha256': {n: O.sha(sink.output / n) for n in (O.ENTRIES, O.STATUS)},
        'quality_gate_clear': False, 'source_values_changed': False, 'label_columns_read': False,
        'streaming': True, 'image_payload_retained': False})


def verify(output: Path) -> None:
    receipt = json.loads((output / O.RECEIPT).read_text())
    status = json.loads((output / O.STATUS).read_text())
    O.require(receipt['guards'] == guards() and set(receipt['sha256']) == {O.ENTRIES, O.STATUS}, 'bounded_saved_guards')
    O.require(all(O.sha(output / n) == h for n, h in receipt['sha256'].items()), 'bounded_saved_sha')
    O.require(status['closed'] is True and status['errors'] == [] and status['rows'] == receipt['rows'], 'bounded_saved_failure')
    O.require(O.saved_counts(output) == (receipt['rows'], receipt['actual_append_count']), 'bounded_saved_counts')
    O.require(all(receipt[k] is False for k in ('quality_gate_clear', 'source_values_changed', 'label_columns_read',
        'image_payload_retained')) and receipt['streaming'] is True, 'bounded_permissions')
