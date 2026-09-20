"""正常・欠測・不明wait・排他保存を実ファイルと実CLIで確認する。"""
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace
from scripts import g3_run_outcome as O


class OutcomeTests(unittest.TestCase):
    def fixture(self) -> tuple[Path, dict]:
        root = O.F.VERIFY / 'g3_repair_2026-09-15_v1/outcome_cpu' / str(uuid.uuid4())
        (root / 'run').mkdir(parents=True)
        O.F.save(root / 'run/ENTRY_RESULT.json', dict(status=O.ENDED, error=None))
        (root / 'run.resources.jsonl').write_text('{"safety_stop":false}\n')
        waited = dict(child_exit_code=0, resource_guard_exit=0, guard_exited_first=False,
                      forced_child_kill=False, supervisor_error=None)
        return root, waited

    def test_normal_and_duplicate(self) -> None:
        root, waited = self.fixture()
        before = (root / 'run/ENTRY_RESULT.json').read_bytes()
        value = O.finish(root, 'run', waited)
        self.assertEqual((value['status'], value['parent_exit_code']), (O.ENDED, 0))
        self.assertFalse(value['quality_gate_clear'])
        with self.assertRaises(FileExistsError):
            O.finish(root, 'run', waited)
        self.assertEqual((root / 'run/ENTRY_RESULT.json').read_bytes(), before)

    def test_failed_cases(self) -> None:
        for case in ('child', 'guard', 'early', 'forced', 'unknown', 'bad_entry', 'bad_resources', 'resource_stop'):
            with self.subTest(case=case):
                root, waited = self.fixture()
                if case == 'child': waited['child_exit_code'] = 137
                if case == 'guard': waited['resource_guard_exit'] = 1
                if case == 'early': waited['guard_exited_first'] = True
                if case == 'forced': waited['forced_child_kill'] = True
                if case == 'unknown': waited = None
                if case == 'bad_entry': (root / 'run/ENTRY_RESULT.json').write_text('{broken')
                if case == 'bad_resources': (root / 'run.resources.jsonl').write_text('{broken')
                if case == 'resource_stop': (root / 'run.resources.jsonl').write_text('{"safety_stop":true}\n')
                result = O.finish(root, 'run', waited)
                self.assertEqual((result['status'], result['parent_exit_code']), (O.FAILED, 1))
                self.assertFalse(result['quality_gate_clear'])

    def test_missing_entry_cli_nonzero(self) -> None:
        root, waited = self.fixture()
        # 未生成の別runを使い、正常fixtureの原票は消さない。
        (root / 'missing').mkdir()
        (root / 'missing.resources.jsonl').write_text('{"safety_stop":true}\n')
        O.F.save(root / 'missing_ACTUAL_WAIT.json', waited | dict(child_exit_code=137, forced_child_kill=True))
        result = subprocess.run([sys.executable, '-m', 'scripts.g3_run_outcome', '--root', str(root),
                                 '--label', 'missing'], capture_output=True, text=True)
        (root / 'CLI.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 1)
        saved = O.F.read(root / 'missing_PARENT_RESULT.json')
        self.assertTrue(saved['artifact_errors'])
        self.assertFalse((root / 'missing/ENTRY_RESULT.json').exists())

    def test_missing_wait_cli(self) -> None:
        root, _ = self.fixture()
        result = subprocess.run([sys.executable, '-m', 'scripts.g3_run_outcome', '--root', str(root),
                                 '--label', 'run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIsNotNone(O.F.read(root / 'run_PARENT_RESULT.json')['wait_error'])

    def test_save_failure_and_destination(self) -> None:
        root, waited = self.fixture()
        with patch.object(O.F, 'save', side_effect=OSError('disk_failed')):
            with self.assertRaisesRegex(OSError, 'disk_failed'):
                O.finish(root, 'run', waited)
        with self.assertRaises(ValueError):
            O.finish(root, '../escape', waited)

    def test_supervisor_connection_failure(self) -> None:
        root, waited = self.fixture()
        calls: list = []
        def run_pair(*args: object) -> dict:
            calls.append(args)
            return waited | dict(child_exit_code=137, forced_child_kill=True)
        value = O.run(root, 'run', ['fixture'], Path('fixture'), api=SimpleNamespace(run_pair=run_pair))
        self.assertEqual(value['parent_exit_code'], 1)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(FileExistsError):
            O.run(root, 'run', ['fixture'], Path('fixture'), api=SimpleNamespace(run_pair=run_pair))
        self.assertEqual(len(calls), 1)

    def test_supervisor_exception(self) -> None:
        root, _ = self.fixture()
        def run_pair(*args: object) -> dict:
            raise RuntimeError('supervisor_unknown')
        value = O.run(root, 'run', ['fixture'], Path('fixture'), api=SimpleNamespace(run_pair=run_pair))
        self.assertEqual(value['parent_exit_code'], 1)
        self.assertFalse(value['wait_verified'])
        self.assertIn('supervisor_unknown', value['wait_error'])


if __name__ == '__main__':
    unittest.main()
