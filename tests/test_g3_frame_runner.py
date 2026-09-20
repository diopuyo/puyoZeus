"""G3入力の時計・終端・例外保存を、認識合格と分離して検査する。"""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from scripts import g3_frame_runner as G

TEST_ROOT = G.VERIFY / 'g3_first_pass_2026-09-14_v1/frame_cpu'


class Capture:
    """read回数と解放を外部から確認できる入力fixture。"""
    def __init__(self, count: int = 8, release_error: bool = False) -> None:
        self.count, self.release_error = count, release_error
        self.position, self.calls, self.released = 0, 0, False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> int:
        return self.position

    def read(self) -> tuple[bool, object]:
        self.calls += 1
        if self.position >= self.count:
            return False, None
        self.position += 1
        return True, np.zeros((2, 3, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True
        if self.release_error:
            raise OSError('fixture_release')


class FrameRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = replace(G.contracts()[0], width=3, height=2,
                              frame_count=8, processing_end_frame_exclusive=7)
        self.output = TEST_ROOT / str(uuid.uuid4())

    def test_all_registered_clocks_and_strides(self) -> None:
        expected = [Fraction(1, 30), Fraction(1, 15), Fraction(1, 30),
                    Fraction(14250, 427073), Fraction(1, 15), Fraction(1, 15)]
        sources = G.contracts()
        self.assertEqual([G.frame_time(s, 2) for s in sources], expected)
        self.assertEqual([G.stride_for(s) for s in sources], [2, 1, 2, 2, 1, 1])
        self.assertEqual([s.processing_end_frame_exclusive for s in sources],
                         [36300, 11878, 15805, 151528, 6760, 14347])

    def test_odd_exclusive_end_and_continuous_decode(self) -> None:
        cap, rows, state = Capture(), [], dict(delivered_count=0)
        G.consume(cap, self.source, 7, lambda f, t, p: rows.append(f), state)
        self.assertEqual(rows, [0, 2, 4, 6])
        self.assertEqual(cap.calls, 7)
        self.assertEqual(state['decoded_count'], 7)

    def test_early_eof_preserves_failure_and_releases(self) -> None:
        cap = Capture(count=2)
        result = G.run_probe(self.source, self.output, lambda _: cap)
        self.assertEqual(result['error']['message'], 'early_eof:2')
        self.assertEqual(result['decoded_count'], 2)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(cap.released)
        saved = json.loads((self.output / 'PROBE_RESULT.json').read_text())
        self.assertEqual(saved, result)
        self.assertFalse((self.output / 'COMPLETE').exists())

    def test_probe_is_separate_and_duplicate_cannot_overwrite(self) -> None:
        cap = Capture()
        result = G.run_probe(self.source, self.output, lambda _: cap)
        original = (self.output / 'PROBE_RESULT.json').read_bytes()
        self.assertEqual(cap.calls, G.PROBE_FRAMES)
        self.assertEqual(result['source']['processing_end_frame_exclusive'], 7)
        self.assertFalse(result['quality_pass'])
        with self.assertRaises(FileExistsError):
            G.run_probe(self.source, self.output, lambda _: Capture())
        self.assertEqual((self.output / 'PROBE_RESULT.json').read_bytes(), original)

    def test_release_failure_does_not_hide_original_error(self) -> None:
        result = G.run_probe(self.source, self.output,
                             lambda _: Capture(count=1, release_error=True))
        self.assertEqual(result['error']['message'], 'early_eof:1')
        self.assertEqual(result['cleanup_errors'][0]['stage'], 'release')
        self.assertFalse(result['release_succeeded'])

    def test_factory_exception_saves_without_claiming_release(self) -> None:
        def broken(_: str) -> Capture:
            raise RuntimeError('factory_failed')
        result = G.run_probe(self.source, self.output, broken)
        self.assertEqual(result['error']['message'], 'factory_failed')
        self.assertFalse(result['release_succeeded'])

    def test_callback_failure_does_not_count_delivery(self) -> None:
        def broken(f: int, t: Fraction, p: object) -> None:
            raise RuntimeError('callback_failed')
        state = dict(delivered_count=0)
        with self.assertRaisesRegex(RuntimeError, 'callback_failed'):
            G.consume(Capture(), self.source, 7, broken, state)
        self.assertEqual(state['delivered_count'], 0)
        self.assertEqual(state['decoded_count'], 1)

    def test_position_and_shape_mismatch_are_rejected(self) -> None:
        cap = Capture()
        cap.position = 1
        with self.assertRaisesRegex(ValueError, 'frame_zero'):
            G.consume(cap, self.source, 4, lambda *a: None, dict(delivered_count=0))
        source = replace(self.source, width=4)
        result = G.run_probe(source, self.output, lambda _: Capture())
        self.assertEqual(result['error']['message'], 'decode_shape_mismatch:0')

    def test_unknown_contract_and_wrong_drive_rejected(self) -> None:
        for changes in [dict(time_base_numerator=0), dict(processing_start_frame=2),
                        dict(processing_end_frame_exclusive=9), dict(width=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                G.validate(replace(self.source, **changes))
        with self.assertRaisesRegex(ValueError, 'verify_drive'):
            G.run_probe(self.source, G.ROOT / 'must_not_create', lambda _: Capture())


if __name__ == '__main__':
    unittest.main()
