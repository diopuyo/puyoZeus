"""初回一巡の独立失敗継続・共有失敗停止・重複拒否を検査する。"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import uuid

from scripts import g3_first_pass as G


class BatchTests(unittest.TestCase):
    def run_fixture(self, resource_stop: bool, defect: str = '') -> tuple[list, dict, list[str]]:
        root = G.VERIFY / 'g3_first_pass_2026-09-14_v1/batch_cpu' / str(uuid.uuid4())
        root.mkdir(parents=True)
        plan = root / 'plan.json'
        plan.write_text('{}')
        output, calls = root / 'batch', []
        def run_pair(command: list[str], guard: str, runner: str, resources: Path) -> dict:
            arm = command[command.index('--arm') + 1]
            calls.append(arm)
            if defect == 'raise':
                raise RuntimeError('fixture_supervisor_failure')
            child = Path(command[command.index('--output') + 1])
            child.mkdir()
            G.save(child / 'ENTRY_RESULT.json', dict(status='FIXTURE_FAILED',
                   error=dict(message='candidate_algorithm_failure')))
            resources.write_text(json.dumps(dict(safety_stop=resource_stop)) + '\n')
            if defect == 'entry':
                (child / 'ENTRY_RESULT.json').write_text('{broken')
            if defect == 'null':
                (child / 'ENTRY_RESULT.json').write_text('{"error":null}')
            if defect == 'resources':
                resources.write_text('{broken')
            if defect == 'wait':
                return {}
            return dict(child_exit_code=1, resource_guard_exit=0, guard_exited_first=False,
                        forced_child_kill=False, supervisor_error=None, source='fixture_no_process')
        argv = ['batch', '--plan', str(plan), '--plan-sha', hashlib.sha256(plan.read_bytes()).hexdigest(),
                '--output', str(output)]
        with patch('sys.argv', argv), patch.object(G, 'supervisor', return_value=N(run_pair=run_pair)):
            self.assertEqual(G.main(), 0)
        return calls, G.read(output / 'FIRST_PASS_RESULT.json'), argv

    def test_algorithm_failure_continues_independent_baseline(self) -> None:
        calls, result, argv = self.run_fixture(False)
        self.assertEqual(calls, ['candidate', 'baseline'])
        self.assertFalse(result['g3_complete'])
        self.assertEqual(len(result['other_sources']), 5)
        with patch('sys.argv', argv), patch.object(G, 'supervisor') as launch:
            with self.assertRaises(FileExistsError):
                G.main()
            launch.assert_not_called()

    def test_resource_stop_prevents_dependent_launch(self) -> None:
        calls, result, _ = self.run_fixture(True)
        self.assertEqual(calls, ['candidate'])
        self.assertEqual(result['video_38'][1]['status'], 'NOT_RUN_SHARED_FAILURE')

    def test_broken_entry_and_null_error_preserve_independent_arm(self) -> None:
        for defect in ('entry', 'null'):
            with self.subTest(defect=defect):
                calls, result, _ = self.run_fixture(False, defect)
                self.assertEqual(calls, ['candidate', 'baseline'])
                self.assertTrue(result['video_38'][0]['wait_verified'])

    def test_unknown_monitor_or_wait_preserves_partial_and_stops(self) -> None:
        for defect in ('resources', 'wait', 'raise'):
            with self.subTest(defect=defect):
                calls, result, argv = self.run_fixture(False, defect)
                self.assertEqual(calls, ['candidate'])
                self.assertEqual(result['video_38'][1]['status'], 'NOT_RUN_SHARED_FAILURE')
                if defect != 'resources':
                    self.assertTrue((Path(argv[-1]) / 'candidate_CONTROL_FAILURE.json').exists())


if __name__ == '__main__':
    unittest.main()
