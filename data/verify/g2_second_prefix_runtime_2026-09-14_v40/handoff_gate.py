"""有限実走の終端直前に新入力の正例を要求。旧M1二採録は互換観測へ分離。"""
import json
import hashlib
import sys
from typing import Any

STRIDE = 2
NAME = 'EARLY_INPUT_LIVE_GATE.json'


def verify_saved(output: Any, last_frame: int) -> dict:
    """入口の早期終了も成功へ抜けないよう、seal/親finalizeの両方で要求する。"""
    raw = (output / NAME).read_bytes()
    packet = json.loads(raw)
    counts = packet['root_probability_counts']
    if (packet['frame'] != last_frame - STRIDE or counts['last_frame'] != packet['frame']
            or type(counts['READY']) is not int or counts['READY'] < 1
            or counts['calls'] != counts['READY'] + counts['HOLD']
            or packet['reference_counts']['REFERENCE_ONLY'] < 1):
        raise ValueError('handoff_gate_saved_positive_or_clock')
    if (packet['history_transferred'] is not True or packet['probability_candidates'] < 1
            or not packet['history_first'] <= packet['history_last'] < packet['frame']
            or packet['original_M1_two_sample_gate_unchanged'] is not False
            or packet['legacy_M1_coverage_observational'] is not True
            or packet['quality_gate_clear'] is not False):
        raise ValueError('handoff_gate_saved_history_or_authority')
    import early_probability_capture as probability
    probability.verify_lifetime(output)
    return dict(frame=packet['frame'], root_READY=counts['READY'],
        source_sha256=hashlib.sha256(raw).hexdigest(), saved_input_gate_verified=True, quality_gate_clear=False)


def check(adapter: Any, bridge: Any, frame: int) -> dict:
    runtime, connection = adapter.CURRENT, adapter.PREVIOUS.CURRENT
    if runtime is None or connection is None: raise ValueError('handoff_gate_runtime_missing')
    history = runtime.require_history()
    if not history.transferred or not history.probability_inputs():
        raise ValueError('handoff_gate_probability_history_missing')
    if history.journal is not bridge.state['atomic_journal_observer'] or history.pipe is not bridge.initial['pipeline']:
        raise ValueError('handoff_gate_history_owner')
    connection.binding.verify(connection.bound, require_instance=True, require_projected_input=True,
        require_reference_sample=True, require_root_probability_sample=True)
    binding = connection.bound['binding']
    if binding['root_probability_counts']['last_frame'] != frame:
        raise ValueError('handoff_gate_input_not_current')
    return dict(frame=frame, history_first=history.frames[0], history_last=history.last_frame,
        history_updates=history.count, history_transferred=True,
        probability_candidates=len(history.probability_inputs()),
        reference_counts=dict(binding['reference_counts']), root_probability_counts=dict(binding['root_probability_counts']),
        original_M1_two_sample_gate_unchanged=False, legacy_M1_coverage_observational=True,
        full_exit_verified=False,
        physical_identity_verified=False, quality_gate_clear=False)


def install(stack: Any, adapter: Any) -> None:
    whole = adapter.A.A.A.V4.A
    original = whole.W.Bridge
    target = adapter.W.LAST - STRIDE
    def bridge(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        previous = value.consumer
        def completed(boundary: Any, frame: int) -> None:
            previous(boundary, frame)
            if frame != target: return
            if boundary is not value: raise ValueError('handoff_gate_bridge_owner')
            try:
                packet = check(adapter, value, frame)
            except BaseException as error:
                try:
                    with (value.state['output'] / 'EARLY_INPUT_LIVE_GATE_FAILURE.json').open('x') as stream:
                        json.dump(dict(frame=frame, error=repr(error), quality_gate_clear=False), stream)
                except BaseException as save_error:
                    print('EARLY_INPUT_GATE_SAVE_ERROR=' + repr(save_error), file=sys.stderr, flush=True)
                raise
            with (value.state['output'] / NAME).open('x') as stream:
                json.dump(packet, stream)
        value.consumer = completed
        return value
    adapter.A.A.A.V4.replace_owned(stack, whole.W, 'Bridge', bridge)
