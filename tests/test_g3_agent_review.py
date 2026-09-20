"""G3 API接続の重複起動・欠測・異常終了をローカル実子で検証する。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from scripts import g3_agent_review as G

TEST_ROOT = G.VERIFY / 'g3_first_pass_2026-09-14_v1' / 'api_cpu'
FIXTURE = r'''
import json,pathlib,sys,time
mode,counter=sys.argv[1:]
with pathlib.Path(counter).open('a') as f: f.write('spawn\n')
pathlib.Path('WORK_PACKET.md').read_text(encoding='utf-8')
def emit(x): print(json.dumps(x),flush=True)
emit({'type':'system','subtype':'init','model':'claude-opus-5' if mode!='wrong_model' else 'wrong'})
if mode=='system_string': emit({'type':'system','subtype':'permission_denied','message':'fixture denied path'})
if mode=='block': time.sleep(20)
if mode=='slow': time.sleep(.5)
call={'file_path':'WORK_PACKET.md'}
if mode=='partial': call['limit']=1
emit({'type':'assistant','message':{'id':'m1','content':[{'type':'tool_use','name':'Read','id':'r1','input':call}]}})
emit({'type':'assistant','message':{'id':'m1','content':[]}})
emit({'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'r1','content':'fixture actual file read'}]}})
if mode=='malformed': print('NOT_JSON',flush=True)
if mode=='retry': emit({'type':'system','subtype':'api_retry','attempt':1})
if mode=='missing_result': sys.exit(7)
if mode=='limit_wait':
 emit({'type':'stream_event','event':{'type':'message_delta','delta':{'stop_reason':'max_tokens'}}})
 time.sleep(2)
 with pathlib.Path(counter).open('a') as f: f.write('continued\n')
if mode=='recovered_cutoff': emit({'type':'stream_event','event':{'type':'message_delta','delta':{'stop_reason':'max_tokens'}}})
if mode!='missing_reason': emit({'type':'stream_event','event':{'type':'message_delta','delta':{'stop_reason':'max_tokens' if mode=='cutoff' else 'end_turn'}}})
r={'type':'result','subtype':'success','is_error':False,'result':'fixture review, not independent API','total_cost_usd':.01}
if mode!='no_usage': r['usage']={'input_tokens':2,'cache_creation_input_tokens':5,'cache_read_input_tokens':7,'output_tokens':11,'output_tokens_details':{'thinking_tokens':3}}
emit(r)
sys.exit(7 if mode=='failure' else 0)
'''


class G3ReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        """テスト原票もDの排他ディレクトリに残す。"""
        self.folder = TEST_ROOT / (self._testMethodName + '_' + uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True)
        self.source = self.folder / 'evidence.txt'
        self.source.write_text('固定入力\n正常対照\n', encoding='utf-8')
        self.counter = self.folder / 'starts.log'
        self.root = self.folder / 'api'
        self.packet = dict(event_id='event01', event='design', unit='g3_test', purpose='CPU契約',
                           completion='実waitと保存', scope={'code': 'fixed'}, delta='new', unresolved=[],
                           evidence=[dict(path=str(self.source), sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                                          first_line=1, last_line=2)])

    def invoke(self, mode: str = 'ok', packet: dict | None = None,
               timeout: float = G.TIMEOUT_SECONDS) -> dict:
        return G.dispatch(packet or self.packet, self.root,
                          argv_override=[sys.executable, '-c', FIXTURE, mode, str(self.counter)], timeout=timeout)

    def count(self) -> int:
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def test_real_cli_system_string_shape_does_not_break_summary(self) -> None:
        result = self.invoke('system_string')
        self.assertEqual(result['status'], 'REVIEW_RETURNED')
        self.assertEqual(result['result']['stop_reasons'], ['end_turn'])

    def test_terminal_cutoff_is_incomplete(self) -> None:
        result = self.invoke('cutoff')
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')
        self.assertEqual(result['result']['stop_reasons'], ['max_tokens'])

    def test_recovered_cutoff_is_retained_but_not_accepted(self) -> None:
        result = self.invoke('recovered_cutoff')
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')
        self.assertEqual(result['result']['stop_reasons'], ['max_tokens', 'end_turn'])

    def test_output_limit_stops_owned_cli_before_continuation(self) -> None:
        result = self.invoke('limit_wait')
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')
        self.assertEqual(self.count(), 1)
        self.assertEqual(result['result']['actual_wait']['error']['message'],
                         'output_limit_detected_stop_owned_cli')
        self.assertEqual(G.process_identity(result['result']['actual_wait']['pid'])['state'], 'absent')

    def test_late_cutoff_cannot_pass_even_if_process_already_exited(self) -> None:
        outcome = self.invoke('recovered_cutoff')
        waited = dict(outcome['result']['actual_wait'], exit_code=0, error=None)
        with patch.object(G, 'save'):
            result = G.summarize(Path(outcome['request']), waited)
        self.assertTrue(result['read_receipts']['packet_read_completed'])
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')

    def test_missing_terminal_reason_is_not_invented(self) -> None:
        result = self.invoke('missing_reason')
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')
        self.assertTrue(result['result']['terminal_reason_missing'])

    def test_pid_reuse_is_distinguished_without_launch(self) -> None:
        saved = dict(process_identity=dict(state='present', start_identity='100'))
        self.assertEqual(G.reconcile_identity(saved, dict(state='present', start_identity='101')), 'PID_REUSED')
        self.assertEqual(G.reconcile_identity(saved, dict(state='present', start_identity='100')), 'SAME_PROCESS')
        self.assertEqual(G.reconcile_identity(saved, dict(state='absent')), 'ABSENT')
        self.assertEqual(G.reconcile_identity(None, dict(state='absent')), 'UNKNOWN')

    def test_summary_exception_is_saved_and_never_resends(self) -> None:
        with patch.object(G, 'summarize', side_effect=AttributeError('fixture_summary_failure')):
            with self.assertRaisesRegex(AttributeError, 'fixture_summary_failure'):
                self.invoke()
        request = next((self.root / 'requests').iterdir())
        self.assertEqual(G.read(request / 'SUMMARY_ERROR.json')['type'], 'AttributeError')
        self.assertTrue(G.read(request / 'WAIT.json')['actual_wait'])
        self.assertEqual(list((self.root / 'slots').glob('*.json')), [])
        self.assertEqual(self.invoke()['status'], 'UNKNOWN_NO_RESEND')
        self.assertEqual(self.count(), 1)

    def test_launch_failure_is_saved_without_counting_unstarted_cli(self) -> None:
        with patch.object(G.subprocess, 'Popen', side_effect=OSError('fixture_spawn_error')):
            result = self.invoke()
        self.assertEqual(result['new_cli_invocations'], 0)
        self.assertEqual(result['result']['actual_wait']['error']['type'], 'OSError')
        self.assertIsNone(result['result']['provider_usage'])
        self.assertEqual(self.invoke()['new_cli_invocations'], 0)

    def test_same_event_reuses_actual_child_and_preserves_usage(self) -> None:
        first = self.invoke()
        second = self.invoke()
        self.assertEqual(self.count(), 1)
        self.assertEqual(first['status'], 'REVIEW_RETURNED')
        self.assertTrue(second['reused'])
        self.assertEqual(first['result']['api_response_count_observed'], 1)
        self.assertEqual(first['result']['provider_usage']['output_tokens'], 11)
        self.assertFalse(first['result']['quality_pass'])
        self.assertEqual(list((self.root / 'slots').glob('*.json')), [])

    def test_same_id_changed_range_is_rejected(self) -> None:
        self.invoke()
        changed = copy.deepcopy(self.packet)
        changed['evidence'][0]['last_line'] = 1
        with self.assertRaisesRegex(ValueError, 'event_content_conflict:existing_key='):
            self.invoke(packet=changed)
        self.assertEqual(self.count(), 1)

    def test_different_id_same_content_is_bound_and_reused(self) -> None:
        self.invoke()
        alias = dict(self.packet, event_id='alias01')
        self.assertTrue(self.invoke(packet=alias)['reused'])
        with self.assertRaisesRegex(ValueError, 'event_content_conflict'):
            self.invoke(packet=dict(alias, delta='changed'))
        self.assertEqual(self.count(), 1)

    def test_concurrent_same_event_launches_only_once(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.invoke('slow'), range(2)))
        self.assertEqual(self.count(), 1)
        self.assertEqual(sum(r['new_cli_invocations'] for r in results), 1)

    def reserve_incomplete(self) -> Path:
        key, _ = G.prepare(self.packet, 'claude-opus-5')
        request = self.root / 'requests' / key
        request.mkdir(parents=True)
        (self.root / 'events').mkdir()
        G.save(self.root / 'events/event01.json', dict(key=key, state='CREATED'))
        G.save(request / 'REQUEST.json', dict(model='claude-opus-5', state='CREATED'))
        return request

    def test_crash_before_pid_save_is_unknown_without_resend(self) -> None:
        self.reserve_incomplete()
        result = self.invoke()
        self.assertEqual(result['status'], 'UNKNOWN_NO_RESEND')
        self.assertEqual(self.count(), 0)

    def test_incomplete_live_pid_is_inspected_without_restart(self) -> None:
        request = self.reserve_incomplete()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
        try:
            identity = G.process_identity(child.pid)
            G.save(request / 'LAUNCH.json', dict(pid=child.pid, process_identity=identity))
            result = self.invoke()
            self.assertEqual(result['current_process_identity'], identity)
            self.assertEqual(result['status'], 'UNKNOWN_NO_RESEND')
            self.assertEqual(self.count(), 0)
        finally:
            G.stop(child)

    def test_child_exit_nonzero_is_not_review_success(self) -> None:
        result = self.invoke('failure')['result']
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')
        self.assertEqual(result['actual_wait']['exit_code'], 7)

    def test_missing_result_preserves_unknown_usage(self) -> None:
        result = self.invoke('missing_result')['result']
        self.assertTrue(result['usage_missing'])
        self.assertIsNone(result['provider_usage'])
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')

    def test_malformed_raw_event_is_saved_as_failure(self) -> None:
        outcome = self.invoke('malformed')
        self.assertEqual(len(outcome['result']['malformed_lines']), 1)
        self.assertEqual(outcome['status'], 'INCOMPLETE_REVIEW')
        self.assertIn('NOT_JSON', (Path(outcome['request']) / 'stdout.jsonl').read_text())

    def test_retry_count_does_not_add_or_invent_tokens(self) -> None:
        result = self.invoke('retry')['result']
        self.assertEqual(result['api_retries_observed'], 1)
        self.assertEqual(result['provider_usage']['input_tokens'], 2)
        self.assertIsNone(result['actual_billed_usd'])

    def test_usage_missing_on_success_is_flagged_not_zero(self) -> None:
        result = self.invoke('no_usage')['result']
        self.assertIsNone(result['provider_usage'])
        self.assertTrue(result['usage_missing'])

    def test_partial_read_does_not_certify_evidence(self) -> None:
        result = self.invoke('partial')['result']
        self.assertFalse(result['read_receipts']['packet_read_completed'])
        self.assertEqual(result['status'], 'INCOMPLETE_REVIEW')

    def test_model_mismatch_is_incomplete(self) -> None:
        self.assertEqual(self.invoke('wrong_model')['status'], 'INCOMPLETE_REVIEW')

    def test_timeout_waits_owned_child_and_releases_slot(self) -> None:
        result = self.invoke('block', timeout=.05)['result']
        self.assertEqual(result['actual_wait']['error']['type'], 'TimeoutExpired')
        self.assertEqual(G.process_identity(result['actual_wait']['pid'])['state'], 'absent')
        self.assertEqual(list((self.root / 'slots').glob('*.json')), [])

    def test_pid_save_failure_stops_actual_child_and_preserves_exception(self) -> None:
        original = G.save
        def fail_launch(path: Path, value: object) -> None:
            if path.name == 'LAUNCH.json':
                raise OSError('fixture save failure')
            original(path, value)
        with patch.object(G, 'save', side_effect=fail_launch):
            result = self.invoke('block')['result']
        self.assertEqual(result['actual_wait']['error']['message'], 'fixture save failure')
        self.assertEqual(G.process_identity(result['actual_wait']['pid'])['state'], 'absent')
        self.assertEqual(list((self.root / 'slots').glob('*.json')), [])

    def test_unsupported_phase_and_evidence_overflow_never_launch(self) -> None:
        with self.assertRaisesRegex(ValueError, 'g3_packet_contract'):
            self.invoke(packet=dict(self.packet, event='g4'))
        self.source.write_text('x' * (G.MAX_EVIDENCE_CHARS + 1), encoding='utf-8')
        self.packet['evidence'][0].update(sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(), last_line=1)
        with self.assertRaisesRegex(ValueError, 'excerpt_too_large'):
            self.invoke()
        self.assertEqual(self.count(), 0)

    def test_unknown_slots_are_not_removed_or_bypassed(self) -> None:
        (self.root / 'slots').mkdir(parents=True)
        for index in range(G.MAX_SLOTS):
            G.save(self.root / 'slots' / f'{index}.json', dict(owner_pid=999999, claim_id='unknown'))
        with self.assertRaisesRegex(RuntimeError, 'claude_slots_full'):
            self.invoke()
        self.assertEqual(len(list((self.root / 'slots').glob('*.json'))), G.MAX_SLOTS)
        self.assertEqual(self.count(), 0)

    def test_output_budget_is_part_of_request_identity(self) -> None:
        before, _ = G.prepare(self.packet, 'claude-fable-5-1')
        with patch.object(G, 'MAX_RESPONSE_TOKENS', 3500):
            after, _ = G.prepare(self.packet, 'claude-fable-5-1')
        self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
