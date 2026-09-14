"""有限メタデータだけを保存し、元frame/call/raw画像を保持しない。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
from typing import Any
import current_call_fixed as F
import side_snapshot as S

KIND = 'conditional_current_capture/v1'
SCOPE_SIZE = 7


def base(item: Any) -> dict[str, Any]:
    scope, frame = item['scope'], item['frame']
    F.require(type(item['token']) is str and bool(item['token']), 'J_token')
    return dict(kind=KIND, frame=scope['frame_idx'], clock=scope['time_sec'], side=scope['side'],
        token=item['token'], code_sha256=hashlib.sha256(frame.f_code.co_code).hexdigest(),
        scope7=None, call_found=None, made=False, make_invoked=None, current=None, error=None, side_at_capture=None,
        stage='capture_return_before_original_cleanup', same_producer_as_outputs=True,
        physical_certified=False, integer_current_permission=False)


def success(module: Any, control: Any, journal: Any, item: Any, returned: Any) -> dict[str, Any]:
    row = base(item)
    frame, scope, pipe = item['frame'], item['scope'], item['pipe']
    F.require(journal.active is None and frame.f_code in journal.codes, 'actual_J_frame')
    F.require(type(returned) is tuple and len(returned) == 2, 'capture_return_shape')
    call, made = returned
    F.require(call is control.calls.get(id(frame)), 'original_call_identity')
    row.update(call_found=call is not None, make_invoked=call is not None, made=made is not None)
    if call is None:
        F.require(made is None, 'made_without_call')
        return row
    view = call['view']
    sm = getattr(pipe, '_sm_' + scope['side'].lower())
    actual = (scope['source_id'], scope['run_id'], item['epoch'], id(pipe), id(sm),
        scope['generation']['reset_epoch'], scope['side'])
    F.require(call['frame'] is frame and view.frame == row['frame'] and view.clock == row['clock'], 'call_clock')
    F.require(type(view.scope) is tuple and len(view.scope) == SCOPE_SIZE and view.scope == actual, 'scope7')
    F.require(scope['pipe_object_id'] == id(pipe) and journal.tracker._pipeline is pipe
        and journal.tracker._machines[row['side']] is sm, 'J_pipe_SM')
    row['scope7'] = list(view.scope)
    if made is not None:
        F.require(type(made) is tuple and len(made) == 2
            and type(made[0]) is module.C.ConditionalCurrent, 'original_current_type')
        row['current'] = asdict(made[0])
        row['side_at_capture'] = S.capture(module.C, made[0], made[1])
    return row


def failed(item: Any, error: BaseException, returned: Any, completed: bool) -> dict[str, Any]:
    try:
        row = base(item)
    except BaseException:
        row = dict(kind=KIND, token=item.get('token'), frame=None, clock=None, side=None,
            code_sha256=None, scope7=None, current=None, side_at_capture=None,
            call_found=None, made=False, make_invoked=None)
    if completed and type(returned) is tuple and len(returned) == 2:
        row.update(call_found=returned[0] is not None, make_invoked=returned[0] is not None,
            made=returned[1] is not None)
    row.update(error=repr(error), stage='capture_failure_before_original_cleanup')
    return row
