"""通知/APIの一回性、原失敗保持、既存supervisor後の接続をCPU検査する。"""
from pathlib import Path
import json
import subprocess
import unittest
from unittest.mock import Mock, patch
import uuid
from scripts import g3_end_event as E
from scripts import g3_run_with_event as W
from tests import test_g3_run_outcome as P


class EventTests(unittest.TestCase):
    def fixture(self, failed: bool = False) -> tuple[Path, dict]:
        root, wait = P.OutcomeTests().fixture()
        if failed:
            wait.update(child_exit_code=137, forced_child_kill=True)
        return root, wait

    def judge(self, work: dict, root: Path) -> dict:
        key, body = E.A.prepare(work, 'claude-fable-5-1')
        self.assertLess(len(work['event_id']), 100)
        self.assertNotIn('NOTIFY_RESULT', body)
        return dict(status='REVIEW_RETURNED', simulated_cpu=True, key=key)

    def test_one_event_and_parent_unchanged(self) -> None:
        for failed in (False, True):
            root, wait = self.fixture(failed)
            E.O.finish(root, 'run', wait)
            parent = root / 'run_PARENT_RESULT.json'
            before = parent.read_bytes()
            notify, judge = Mock(return_value=dict(confirmed=True)), Mock(side_effect=self.judge)
            first = E.process(root, 'run', notify=notify, judge=judge)
            again = E.process(root, 'run', notify=notify, judge=judge)
            self.assertEqual(first['status'], 'EVENT_HANDLED_NOT_G3_PASS')
            self.assertEqual(first['parent_exit_code'], int(failed))
            self.assertTrue(again['reused'])
            self.assertEqual((notify.call_count, judge.call_count), (1, 1))
            self.assertEqual(parent.read_bytes(), before)
            self.assertFalse(first['quality_gate_clear'])

    def test_notification_failure_does_not_block_api_or_retry(self) -> None:
        root, wait = self.fixture(True)
        E.O.finish(root, 'run', wait)
        notify = Mock(side_effect=RuntimeError('delivery_failed'))
        judge = Mock(side_effect=self.judge)
        value = E.process(root, 'run', notify=notify, judge=judge)
        self.assertEqual(value['status'], 'FAILED')
        self.assertIn('delivery_failed', value['notification_error'])
        E.process(root, 'run', notify=notify, judge=judge)
        self.assertEqual((notify.call_count, judge.call_count), (1, 1))

    def test_unknown_process_reconciled_without_resend(self) -> None:
        root, wait = self.fixture()
        E.O.finish(root, 'run', wait)
        notify, judge = Mock(side_effect=SystemExit('interrupted')), Mock()
        with self.assertRaises(SystemExit):
            E.process(root, 'run', notify=notify, judge=judge)
        value = E.process(root, 'run', notify=notify, judge=judge)
        self.assertEqual(value['status'], 'UNKNOWN_NO_RESEND')
        self.assertEqual(value['process_match'], 'SAME_PROCESS')
        self.assertEqual((notify.call_count, judge.call_count), (1, 0))

    def test_api_failure_is_saved_once(self) -> None:
        root, wait = self.fixture()
        E.O.finish(root, 'run', wait)
        notify, judge = Mock(return_value=dict(confirmed=True)), Mock(side_effect=TimeoutError('unknown_API'))
        value = E.process(root, 'run', notify=notify, judge=judge)
        self.assertIn('unknown_API', value['api_error'])
        self.assertEqual(value['status'], 'FAILED')
        E.process(root, 'run', notify=notify, judge=judge)
        self.assertEqual(judge.call_count, 1)

    def test_changed_parent_does_not_send_mixed_evidence(self) -> None:
        root, wait = self.fixture()
        E.O.finish(root, 'run', wait)
        def notify(*args: object) -> dict:
            (root / 'run_PARENT_RESULT.json').write_text('{}')
            return dict(confirmed=True)
        judge = Mock()
        value = E.process(root, 'run', notify=notify, judge=judge)
        self.assertEqual(value['status'], 'FAILED')
        self.assertIn('parent_changed_before_api', value['api_error'])
        self.assertEqual(judge.call_count, 0)

    def test_long_label_and_errors_fit_packet(self) -> None:
        root, wait = self.fixture(True)
        name = 'x' * 90
        E.O.finish(root, name, wait)
        parent = E.O.destination(root, name)
        value = E.A.read(parent)
        value.update(artifact_errors=['原例外' * 10000], audit_error='X' * 10000, wait_error='Y' * 10000)
        parent.write_text(json.dumps(value))
        result = E.process(root, name, notify=lambda *a: dict(confirmed=True), judge=self.judge)
        self.assertEqual(result['status'], 'EVENT_HANDLED_NOT_G3_PASS')
        self.assertEqual(E.A.read(parent)['artifact_errors'], value['artifact_errors'])

    def test_run_keeps_child_failure_even_when_event_succeeds(self) -> None:
        for failed, relay_failed in ((False, False), (True, False), (False, True)):
            root, wait = self.fixture(failed)
            supervisor = Mock()
            supervisor.run_pair.return_value = wait
            relay = Mock(side_effect=RuntimeError('interop_failed')) if relay_failed else Mock(
                return_value=dict(status='EVENT_HANDLED_NOT_G3_PASS'))
            value = W.run(root, 'run', ['cpu_fixture'], Path('runner'), Path('windows_python'),
                          supervisor=supervisor, relay_call=relay)
            self.assertEqual(value['exit_code'], int(failed or relay_failed))
            self.assertEqual(value['parent']['parent_exit_code'], int(failed))
            self.assertEqual((supervisor.run_pair.call_count, relay.call_count), (1, 1))
            self.assertTrue((root / 'run_RUN_EVENT_RESULT.json').is_file())
            with self.assertRaises(FileExistsError):
                W.run(root, 'run', [], Path('runner'), Path('windows_python'), supervisor=supervisor, relay_call=relay)

    @staticmethod
    def windows_path(path: Path) -> str:
        """powershell.exe へ渡せる形にする (2026-09-17、W45)。

        WSL の python から起動すると `/mnt/c/...` のまま渡ってしまい、
        PowerShell は「そのファイルは存在しません」と言って**終了コード0**で終わる。
        つまり通知スクリプトは一度も動かないまま、
        「非0のはず」の検査だけが落ちていた。
        逆に PowerShell が起動失敗で非0を返す環境なら、
        **中身を一度も動かさずに合格してしまう**。どちらも測っていない。
        """
        text = str(path)
        if not text.startswith('/'):
            return text
        done = subprocess.run(['wslpath', '-w', text], capture_output=True, timeout=10)
        if done.returncode != 0:
            raise unittest.SkipTest('wslpath で Windows のパスへ変換できない')
        return done.stdout.decode().strip()

    def test_real_notification_duplicate_and_unknown_claim_no_ui(self) -> None:
        self.assertTrue(E.NOTIFIER.read_bytes().startswith(b'\xef\xbb\xbf'))
        script = self.windows_path(E.NOTIFIER)
        base = E.A.VERIFY / 'g3_repair_2026-09-15_v1/notification_delivery_cpu_v2'
        before = (base / 'NOTIFY_RESULT.json').read_bytes()
        root = E.A.VERIFY / 'g3_repair_2026-09-15_v1/notify_claim_cpu' / uuid.uuid4().hex
        root.mkdir(parents=True)
        E.A.save(root / 'NOTIFY_REQUEST.json', dict(schema='g3-local-notification/v1', title='CPU', body='CPU'))
        E.A.save(root / 'NOTIFY_STARTED.json', dict(pid=0, simulated_interruption=True))
        # base は受領票があるので重複で、root は開始票があるので二重取得で止まる。
        # どちらも画面へ出す前に落ちるので、通知は表示されない。
        for folder, token in ((base, 'g3_notify_duplicate'), (root, 'NOTIFY_STARTED.json')):
            result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-File',
                script, '-RequestPath', self.windows_path(folder / 'NOTIFY_REQUEST.json')],
                capture_output=True, timeout=25)
            text = (result.stdout + result.stderr).decode('utf-8', 'replace')
            (root / (folder.name + '.log')).write_bytes(result.stdout + result.stderr)
            # 起動できずに落ちたのを「拒否できた」と読まないため、
            # 狙った停止理由が出ていることまで確かめる。
            self.assertIn(token, text)
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual((base / 'NOTIFY_RESULT.json').read_bytes(), before)
        self.assertFalse((root / 'NOTIFY_RESULT.json').exists())


if __name__ == '__main__':
    unittest.main()
