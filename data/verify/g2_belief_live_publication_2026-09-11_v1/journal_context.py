"""両sideの原J完了を同updateへ結合。旧NEXT票や整数currentを捏造しない。"""
from __future__ import annotations

from typing import Any

SIDES = ('1P', '2P')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('belief_publication:' + reason)


def join(row: dict[str, Any], steps: list[dict[str, Any]], scope: tuple[Any, ...]) -> tuple[str, str]:
    require(len(steps) == 2 and [step['side'] for step in steps] == list(SIDES), 'two_ordered_J')
    require(row['update']['returned'] is True and row['update']['exception'] is None, 'update_return')
    require((row['source_id'], row['run_id']) == scope[:2], 'context_scope')
    for side, step in zip(SIDES, steps, strict=True):
        require(step['kind'] == 'step' and step['exception'] is None, 'J_completion')
        require((step['source_id'], step['run_id'], step['pipe_object_id'])
                == (scope[0], scope[1], scope[3]), 'J_scope')
        require(step['frame_idx'] == row['frame_idx'] == row['update']['returned_frame_idx']
                and step['time_sec'] == row['time_sec'], 'J_clock')
        generation = row['generation']['after']['value'][side]
        require(step['generation'] == step['generation_after'] == generation, 'J_generation')
        require(type(step['software_reset']) is int and step['software_reset'] >= 0, 'J_reset')
        require(type(step['token']) is str and step['token'].startswith('step:'), 'J_token')
    require(steps[0]['software_reset'] == scope[2]
            and steps[0]['generation']['reset_epoch'] == scope[5] and scope[-1] == SIDES[0], 'basis_scope')
    tokens = tuple(step['token'] for step in steps)
    ordinals = [token.removeprefix('step:') for token in tokens]
    require(all(text.isascii() and text.isdigit() and str(int(text)) == text for text in ordinals), 'J_ordinal')
    require(int(ordinals[1]) == int(ordinals[0]) + 1, 'J_consecutive_tokens')
    return tokens
