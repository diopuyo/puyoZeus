"""既定OFFの経路と、既存Referenceのキャッシュ・色変換を検証する。"""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from src.ama_rust_reference import NativeReference, NativeTransport
from src.phase_j.live_ama_engine import AmaEngine, configured_rust_threads


class RustReferenceTests(unittest.TestCase):
    def test_default_off_and_explicit_override(self) -> None:
        with patch.dict(os.environ, {'PUYO_OBS_AMA_RUST_THREADS': 'bad'}, clear=True):
            self.assertEqual(configured_rust_threads(), 0)
        with patch.dict(os.environ, {'PUYO_OBS_AMA_RUST': '1', 'PUYO_OBS_AMA_RUST_THREADS': '3'}, clear=True):
            self.assertEqual(configured_rust_threads(), 3)
            self.assertEqual(configured_rust_threads(0), 0)
        self.assertRaises(ValueError, configured_rust_threads, -1)
        self.assertRaises(ValueError, NativeTransport, 0)

    def test_reference_cache_and_color_mapping_are_reused(self) -> None:
        calls = []

        def native(text: str, threads: int) -> str:
            calls.append((json.loads(text), threads))
            return json.dumps(dict(candidates=[], expanded_nodes=4,
                initial=dict(board=[1] + [0]*77, scores=[], dead=False)))

        ref = NativeReference(3, native_call=native)
        args = ([0]*78, [[4, 3], [2, 1], [4, 1]], (4, 3, 2, 1))
        first, second = ref.think(*args), ref.think(*args)
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        self.assertEqual(first['initial']['board'][0], 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]['visible'][0], [1, 2])
        self.assertEqual(calls[0][1], 3)
        ref.close()
        self.assertTrue(ref.process.closed)

    def test_flag_off_uses_original_reference(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'read_text', return_value='{}'):
            with patch('src.phase_j.live_ama_engine.Reference') as old, patch.object(Path, 'is_file', return_value=True):
                engine = AmaEngine(Path('old.exe'), Path('model.json'))
                self.assertEqual(old.call_count, 2)
                old.assert_called_with(Path('old.exe'))
                self.assertEqual(engine.rust_threads, 0)

    def test_flag_on_uses_native_reference(self) -> None:
        with patch.object(Path, 'read_text', return_value='{}'):
            with patch('src.ama_rust_reference.NativeReference') as native:
                with patch('src.phase_j.live_ama_engine.Reference') as old:
                    engine = AmaEngine(Path('old.exe'), Path('model.json'), rust_threads=2)
                    self.assertEqual(native.call_count, 2)
                    native.assert_called_with(2)
                    old.assert_not_called()
                    self.assertEqual(engine.rust_threads, 2)


if __name__ == '__main__':
    unittest.main()
