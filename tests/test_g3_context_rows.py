"""原Recorder/原coverage/原verifyで保持削減の同値性と失敗を検査する。"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N
import tracemalloc
import unittest
import uuid
from typing import Any

from scripts import g3_context_rows as C
from scripts import g3_context_entry as E

ROOT = E.G.VERIFY / 'g3_repair_2026-09-15_v1'
COUNT = 128
MAX_RETAINED = 1024 * 1024


def original() -> Any:
    spec = importlib.util.spec_from_file_location('context_rows_cpu_original', E.SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace(stack: ExitStack, owner: Any, name: str, value: Any) -> None:
    old = getattr(owner, name)
    stack.callback(setattr, owner, name, old)
    setattr(owner, name, value)


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out = ROOT / 'context_rows_cpu' / uuid.uuid4().hex
        self.out.mkdir(parents=True)
        self.module = original()
        with (ROOT / 'video38_current_gpu_v5/provisional_context.jsonl').open(encoding='utf-8') as stream:
            self.row = json.loads(next(stream))

    def record(self, name: str, bounded: bool, count: int = 3) -> Any:
        folder = self.out / name
        folder.mkdir()
        module = self.module
        frames = list(range(0, count * 2, 2))
        pairs = [(f, side) for f in frames for side in module.SIDES]
        state = dict(output=folder, hidden_probability_observer=N(expected=pairs,
            source_id=self.row['source_id'], run_id=self.row['run_id']),
            current_scope_sink=N(expected=pairs), provisional_current_connection=N(), tracker=N())
        rec = module.Recorder(N(), state, N(), frames)
        if bounded:
            rec.rows = C.ContextRows(folder / module.SIDECAR)
        rec.installed = True
        for index, frame in enumerate(frames):
            row = copy.deepcopy(self.row)
            row.update(frame_idx=frame, available_frame=frame, time_sec=frame / 60,
                       capture_token=f'update:{index}')
            row['update'].update(frame_idx=frame, returned_frame_idx=frame,
                time_sec=frame / 60, returned_time_sec=frame / 60, call_index=index)
            rec.active = dict(row=row)
            rec.end(None)
        rec.close()
        return rec

    def test_original_end_coverage_finish_verify_equivalence(self) -> None:
        left, right = self.record('list', False), self.record('disk', True)
        m = self.module
        self.assertEqual((left.output / m.SIDECAR).read_bytes(), (right.output / m.SIDECAR).read_bytes())
        self.assertEqual(left.rows, list(right.rows))
        self.assertEqual(left.rows[-1], right.rows[-1])
        a = m.finish({m.STATE_KEY: left})
        b = m.finish({m.STATE_KEY: right})
        self.assertEqual(a, b)
        self.assertEqual(m.verify(right.output, expected_frames=right.expected),
                         C.verify(m, right.output, expected_frames=right.expected))
        for field, bad in [('update_count', 99), ('quality_gate_clear', True), ('source_id', 'wrong')]:
            changed = dict(b, **{field: bad})
            (right.output / m.RECEIPT).write_text(json.dumps(changed))
            for verifier in (m.verify, lambda out, **kw: C.verify(m, out, **kw)):
                with self.assertRaises(ValueError):
                    verifier(right.output, expected_frames=right.expected)

    def test_newlines_nonascii_and_unknown_access(self) -> None:
        for newline in ('\n', '\r\n'):
            path = self.out / ('lf' if newline == '\n' else 'crlf')
            path.touch()
            rows = C.ContextRows(path)
            self.assertFalse(rows)
            with self.assertRaises(IndexError):
                rows[-1]
            value = dict(text='ぷよ🙂', number=1)
            with path.open('a', encoding='utf-8', newline=newline) as f:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
            rows.append(value)
            self.assertEqual([value], list(rows))
            self.assertEqual([value], list(C.saved_rows(path)))
            for index in (0, slice(None), -2, True):
                with self.assertRaises(IndexError):
                    rows[index]
            with self.assertRaises(ValueError):
                C.ContextRows(path)

    def test_saved_damage_and_append_failures(self) -> None:
        for damage in ('truncated', 'changed', 'extra', 'missing'):
            rec = self.record(damage, True)
            path = rec.rows.path
            raw = path.read_bytes()
            if damage == 'truncated': path.write_bytes(raw[:-1])
            elif damage == 'changed': path.write_bytes(raw.replace(b'CAPTURED', b'BROKEN__', 1))
            elif damage == 'extra': path.write_bytes(raw + b'{}\n')
            else: path.unlink()
            with self.assertRaises((ValueError, OSError)):
                list(rec.rows)
        for raw, row in ((b'{}', {}), (b'{}\n{}\n', {}), (b'{}\n', {'bad': 1})):
            path = self.out / uuid.uuid4().hex
            path.touch()
            rows = C.ContextRows(path)
            path.write_bytes(raw)
            with self.assertRaises(ValueError): rows.append(row)
            self.assertEqual(len(rows), 0)

    def test_mutation_and_concurrent_append_are_rejected(self) -> None:
        rec = self.record('mutated', True)
        rec.rows[-1]['frame_idx'] = -1
        with self.assertRaisesRegex(ValueError, 'latest_mutated'): list(rec.rows)
        rec = self.record('concurrent', True)
        iterator = iter(rec.rows)
        next(iterator)
        value = dict(other=True)
        with rec.rows.path.open('ab') as f: f.write(b'{"other": true}\n')
        rec.rows.append(value)
        with self.assertRaises(ValueError): list(iterator)

    def test_actual_constructor_restore_and_failure_save(self) -> None:
        module = self.module
        originals = (module.Recorder.__init__, module.sha, module.verify)
        with ExitStack() as stack:
            E.bind_context(stack, module, replace)
            with self.assertRaises(ValueError): E.bind_context(stack, module, replace)
            rec = self.record('installed', False)
            self.assertIsInstance(rec.rows, C.ContextRows)
            module.finish({module.STATE_KEY: rec})
            module.verify(rec.output, expected_frames=rec.expected)
        self.assertEqual(originals, (module.Recorder.__init__, module.sha, module.verify))
        receipt = json.loads((rec.output / 'G3_CONTEXT_RETENTION_END.json').read_text())
        self.assertTrue(receipt['constructor_restored'] and receipt['stream_closed'])
        self.assertEqual(receipt['retained_rows'], 1)
        self.assertEqual(receipt['count'], 3)

    def test_bounded_retention_effect_on_original_end(self) -> None:
        path = self.out / 'REPEATED_INPUT_CPU.jsonl'
        path.touch()
        rec = self.module.Recorder.__new__(self.module.Recorder)
        rec.rows = C.ContextRows(path)
        with path.open('a', encoding='utf-8') as stream:
            rec.stream = stream
            tracemalloc.start()
            try:
                for _ in range(COUNT):
                    rec.active = dict(row=self.row)
                    rec.end(None)
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(len(rec.rows), COUNT)
        self.assertLess(current, MAX_RETAINED)
        self.assertEqual(list(rec.rows), [self.row] * COUNT)
        result = dict(artificial_repeated_source_row=True, count=COUNT,
            traced_current_bytes=current, traced_peak_bytes=peak, limit_bytes=MAX_RETAINED,
            original_probe=str(ROOT / 'context_retention_probe/RESULT.json'), quality_gate_clear=False)
        (self.out / 'MEMORY_RESULT.json').write_text(json.dumps(result))
        print(json.dumps(dict(memory_result=str(self.out / 'MEMORY_RESULT.json'), **result)))


if __name__ == '__main__':
    unittest.main()
