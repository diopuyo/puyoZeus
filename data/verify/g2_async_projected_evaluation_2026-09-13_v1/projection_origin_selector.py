"""診断対象の側・原J・起点同一性を固定し、曖昧な選択を拒否する。"""
from typing import Any


def selected_origin(step: dict[str, Any], side: str, token: str, object_id: int) -> dict:
    if step.get('side') != side or step.get('token') != token:
        raise ValueError('projection_target_call')
    origins = [event['active_origin'] for event in step['events']
               if event.get('active_origin') is not None]
    identities = {(origin['object_id'], origin['trigger_sec']) for origin in origins}
    if len(identities) != 1 or origins[0]['object_id'] != object_id:
        raise ValueError('projection_target_origin')
    first = origins[0]
    if first.get('before_board') is None or any(
            origin['before_board'] != first['before_board'] for origin in origins):
        raise ValueError('projection_origin_mutated')
    return first
