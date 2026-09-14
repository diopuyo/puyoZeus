"""原second_observationを早期scopeで借りる。PB分布・基準・会計は変更しない。"""
import json
import sys
from typing import Any

SIDE = '2P'
SCOPE_KEYS = ('source_id', 'run_id', 'pipe_object_id', 'software_reset', 'frame_idx', 'time_sec', 'side')


def require(value: bool, reason: str) -> None:
    if not value: raise ValueError('early_probability:' + reason)


class Capture:
    def __init__(self, stack: Any, journal: Any, state: dict, observation: Any, replace: Any, *, hook_stack: Any = None) -> None:
        observer, receiver = state['hidden_probability_observer'], state['postcommit_current_receiver']
        require(observer.installed and not observer.closed and not observer.failures, 'observer_lifetime')
        require(receiver.journal is journal and receiver.rec is state['provisional_context_observer']
                and not receiver.closed and not receiver.errors, 'receiver_owner_or_lifetime')
        original = observation.capture
        require(original.__globals__ is vars(observation) and original.__name__ == 'capture', 'capture_owner')
        self.journal, self.state, self.observation = journal, state, observation
        self.closed, self.evidence = False, None
        self.deferred = hook_stack is not None
        self.original_complete, self.original_capture = journal.complete_step, original
        self.output = state['output'] if self.deferred else None
        self.hook_receipt = None
        self.expected = tuple(observer.expected)
        selected = frozenset(self.expected)
        require(len(selected) == len(self.expected), 'duplicate_expected')
        stack.callback(self.close)
        owner = hook_stack if self.deferred else stack
        if self.deferred: owner.push(self.release)
        def capture(journal: Any, state: dict, item: dict, result: Any, error: Any,
                    registered_scope: Any = None) -> dict:
            require(journal is self.journal and state is self.state, 'capture_lifetime_or_owner')
            if self.closed:
                require(self.deferred, 'capture_closed')
                return dict(frame=item['scope']['frame_idx'], journal_token=item['token'], side=SIDE,
                    hold_reason='early_capture_sealed', basis_registered=False, quality_gate_clear=False)
            require(tuple(observer.expected) == self.expected, 'expected_changed')
            scope = item['scope']
            if (scope['frame_idx'], scope['side']) not in selected:
                return dict(frame=scope['frame_idx'], journal_token=item['token'], side=scope['side'],
                    hold_reason='outside_original_PB_window', basis_registered=False, quality_gate_clear=False)
            return original(journal, state, item, result, error, registered_scope)
        replace(owner, observation, 'capture', capture)
        self.evidence = observation.install(owner, journal, state)

    def snapshot(self, step: dict) -> dict:
        require(not self.closed and self.evidence is not None and not self.evidence.closed
                and self.evidence.error is None, 'snapshot_lifetime')
        value = self.evidence.latest
        require(value is not None and step['side'] == SIDE, 'current_pair')
        side = value.get('side') if value.get('hold_reason') else value['scope'][-1]
        require(side == SIDE and value['frame'] == step['frame_idx']
                and value['journal_token'] == step['token'], 'current_pair')
        if not value.get('hold_reason'):
            expected = [step['source_id'], step['run_id'], step['software_reset'], id(self.journal.pipe),
                id(self.journal.pipe._sm_2p), step['generation']['reset_epoch'], SIDE]
            require(value['scope'] == expected and step['pipe_object_id'] == id(self.journal.pipe), 'scope')
            require(value['clock'] == step['time_sec'] and value['original_code_sha256'] == step['code_sha256'], 'clock_or_code')
        result = dict(value, side=side, journal_scope={key: step[key] for key in SCOPE_KEYS},
            generation=step['generation'], generation_after=step['generation_after'],
            in_step_generation_changed=step['generation'] != step['generation_after'])
        return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))

    def close(self) -> None:
        self.closed = True
        if not self.deferred: self.clear()

    def clear(self) -> None:
        self.journal = self.state = self.observation = self.evidence = None
        self.original_complete = self.original_capture = None
        self.expected = ()

    def release(self, kind: Any, body: Any, trace: Any) -> bool:
        evidence = self.evidence
        receipt = dict(sampling_closed=self.closed, hook_closed=evidence is not None and evidence.closed,
            hook_error=None if evidence is None or evidence.error is None else repr(evidence.error),
            complete_restored=self.journal.complete_step == self.original_complete,
            capture_restored=self.observation.capture is self.original_capture,
            original_error=None if body is None else repr(body), quality_gate_clear=False)
        valid = (receipt['sampling_closed'] and receipt['hook_closed'] and receipt['hook_error'] is None
                 and receipt['complete_restored'] and receipt['capture_restored'])
        self.closed = True
        self.clear()
        receipt['references_released'] = self.journal is self.state is self.observation is self.evidence is None
        self.hook_receipt = receipt
        try:
            with (self.output / 'EARLY_PROBABILITY_LIFETIME.json').open('x') as stream:
                json.dump(receipt, stream, allow_nan=False)
        except BaseException as error:
            if body is None: raise
            print('EARLY_PROBABILITY_RELEASE_SAVE_ERROR=' + repr(error), file=sys.stderr, flush=True)
        if body is None: require(valid, 'hook_release_failed')
        return False

    @staticmethod
    def is_candidate(value: dict) -> bool:
        return candidate(value)


def candidate(value: dict) -> bool:
    """保存候補であり基準登録許可ではない。効果窓等の原条件は票に保持する。"""
    return (not value.get('hold_reason') and value.get('state') == 'stable'
            and value.get('pb_holds') == [] and value.get('pb_errors') == []
            and not value['in_step_generation_changed'])


def verify_lifetime(output: Any) -> dict:
    """通常終了だけを検収する。取得停止と物理hook解放の両方が必要。"""
    value = json.loads((output / 'EARLY_PROBABILITY_LIFETIME.json').read_bytes())
    flags = ('sampling_closed', 'hook_closed', 'complete_restored', 'capture_restored', 'references_released')
    require(all(value[key] is True for key in flags) and value['hook_error'] is None
            and value['original_error'] is None and value['quality_gate_clear'] is False, 'lifetime_not_normal')
    return value
