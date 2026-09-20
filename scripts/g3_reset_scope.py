"""G3私有観測moduleの同step resetを欠測化する。元ファイルは変更しない。"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

SOURCE_SHA = 'a0bccf05468f36eec8b2ab69dbffcfe4b1bf0f86a6b14a7b5d905d86e6184dfc'
MARKER = "    C.require(local['sm'] is sm and local['frame_idx'] == scope['frame_idx']\n"
INSERT = '''    C.require(local['sm'] is sm and local['frame_idx'] == scope['frame_idx']
        and local['time_sec'] == scope['time_sec'], 'g3_reset_input_identity')
    reset_scope = journal.scope(pipe, SIDE, scope['frame_idx'], scope['time_sec'])
    reset_epoch = journal.epoch(pipe, SIDE)
    if sm.context.frame_idx/60 != scope['time_sec'] and (
            reset_scope['generation']['reset_epoch'] != scope['generation']['reset_epoch'] or reset_epoch != item['epoch']):
        C.require(all(reset_scope[k] == scope[k] for k in
            ('source_id', 'run_id', 'side', 'pipe_object_id', 'frame_idx', 'time_sec'))
            and reset_scope['generation']['reset_epoch'] >= scope['generation']['reset_epoch']
            and reset_epoch >= item['epoch'], 'g3_reset_identity_or_reversal')
        C.require(sm.context.frame_idx == 0, 'g3_reset_clock_not_initial')
        return dict(frame=scope['frame_idx'], journal_token=item['token'], side=SIDE,
            hold_reason='generation_changed', basis_registered=False, quality_gate_clear=False,
            observed_sm_frame=sm.context.frame_idx, input_frame=scope['frame_idx'],
            reset_epoch_before=scope['generation']['reset_epoch'],
            reset_epoch_after=reset_scope['generation']['reset_epoch'],
            software_epoch_before=item['epoch'], software_epoch_after=reset_epoch)
'''


def generated(module: Any) -> tuple[Callable, dict]:
    """実私有moduleのglobalsを維持し、localsを分けて属性の早期上書きを避ける。"""
    path = Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('g3_reset_source_changed')
    original = module.capture
    if original.__globals__ is not vars(module) or original.__name__ != 'capture':
        raise ValueError('g3_reset_original_owner')
    source = inspect.getsource(original)
    if source.count(MARKER) != 1:
        raise ValueError('g3_reset_transform_marker')
    transformed = source.replace(MARKER, INSERT + MARKER)
    namespace: dict[str, Any] = {}
    exec(compile(transformed, str(Path(__file__).resolve()), 'exec'), vars(module), namespace)
    value = namespace['capture']
    assert module.capture is original and value.__globals__ is vars(module)
    return value, dict(original_sha256=SOURCE_SHA,
                       generated_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                       generated_source=transformed, quality_gate_clear=False)


def install(stack: Any, module: Any, replace: Callable) -> dict:
    """Capture初期化前に同moduleへ置き、後続hook復元後に元関数へ戻す。"""
    value, receipt = generated(module)
    replace(stack, module, 'capture', value)
    return receipt
