"""計測・電源変更を起動せず、門番・復元・集計窓を検査する。"""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'packaging'))
import live_gate_coef as gate
import live_gate_power as power
from live_gate_analysis import summary, ratios, delays
from live_gate_win import masks


class GateTests(unittest.TestCase):
    def test_preview_and_check_never_execute(self) -> None:
        for argv in (['gate'], ['gate', 'CHECK'], ['gate', 'RESTORE'],
                     ['gate', 'CHECK', '--confirm', gate.CONFIRM]):
            with patch.object(sys, 'argv', argv), patch.object(gate, 'pending_recovery'), \
                    patch.object(gate, 'preflight', return_value={}), patch.object(gate, 'execute') as execute:
                gate.main()
                execute.assert_not_called()

    def test_noninteractive_run_rejected(self) -> None:
        with patch.object(sys, 'argv', ['gate', gate.RUN_WORD]), patch.object(gate, 'pending_recovery'), \
                patch.object(gate, 'preflight', return_value={}), patch.object(sys.stdin, 'isatty', return_value=False), \
                patch.object(gate, 'execute') as execute:
            with self.assertRaises(RuntimeError):
                gate.main()
            execute.assert_not_called()

    def test_automation_needs_both_explicit_words(self) -> None:
        with patch.object(sys, 'argv', ['gate', gate.RUN_WORD, '--confirm', gate.CONFIRM]), \
                patch.object(gate, 'pending_recovery'), patch.object(gate, 'preflight', return_value={}), \
                patch.object(gate, 'execute') as execute:
            gate.main()
            execute.assert_called_once_with({})

    def test_topology_uses_four_physical_performance_cores(self) -> None:
        rows = [dict(group=0, logical=i, core=i//2 if i < 12 else i, efficiency=int(i < 12))
                for i in range(16)]
        self.assertEqual(masks(rows), (65535, 255))
        with self.assertRaises(ValueError):
            masks(rows[:4])

    def test_windows_and_ratios(self) -> None:
        normal = summary([(0., 10.), (1199., 20.), (1200., 9999.), (3000., 20000.)], 3300.)
        low = summary([(0., 20.), (1199., 40.), (1200., 19998.), (3000., 40000.)], 3300.)
        self.assertEqual(normal['first20']['n'], 2)
        self.assertEqual(normal['full']['maximum'], 20000.)
        self.assertEqual(ratios(low, normal), dict(p50=2., p95=2., p99=2.))

    def test_publication_done_and_origin(self) -> None:
        rows = [dict(kind='result', accepted=True, game=1, frame=5, captured_at=10., t_sec=2.),
            dict(kind='publication', game=1, frame=6, state_frame=5, captured_at=11., t_sec=3.,
                 published_at=11.1, monotonic=11.5)]
        with patch('live_gate_analysis.records', side_effect=lambda _: iter(rows)):
            result = delays(Path('unused'))
        self.assertEqual(result, dict(latest_capture=[(3., 500.)], b3_origin=[(2., 1500.)]))

    def test_restore_verifies_original_before_delete(self) -> None:
        state = dict(original='old', temporary='temporary', original_query='unchanged', restored=False)
        calls = []
        def fake_power(*args: str) -> str:
            calls.append(args)
            return 'unchanged' if args[0] == '/qh' else 'temporary'
        with patch.object(Path, 'read_text', return_value=json.dumps(state)), patch.object(Path, 'touch'), \
                patch.object(power, 'power', side_effect=fake_power), patch.object(power, 'active', return_value='old'), \
                patch.object(power, 'write') as write, patch.object(power.winreg, 'CreateKey', return_value=nullcontext(1)), \
                patch.object(power.winreg, 'DeleteValue'):
            power.restore(Path('unused'))
        self.assertTrue(write.call_args.args[1]['restored'])
        self.assertLess(calls.index(('/qh', 'old')), calls.index(('/delete', 'temporary')))

    def test_restore_failure_stays_pending(self) -> None:
        state = dict(original='old', temporary='temporary', original_query='unchanged', restored=False)
        with patch.object(Path, 'read_text', return_value=json.dumps(state)), patch.object(Path, 'touch'), \
                patch.object(power, 'power', side_effect=RuntimeError('failed')), \
                patch.object(power, 'write') as write, patch.object(power.winreg, 'DeleteValue') as delete:
            with self.assertRaises(RuntimeError):
                power.restore(Path('unused'))
        self.assertFalse(write.call_args.args[1]['restored'])
        delete.assert_not_called()


if __name__ == '__main__':
    unittest.main()
