"""retR v2の既定OFFの旗を親設定からEX計算子へ渡す。"""
from __future__ import annotations

import argparse
import os
from typing import Any

FLAGS = (
    ('retr_move050', 'PUYO_LIVE_24H_RETR_MOVE050', 'PUYO_RETR_MOVE_TIME_TOP50'),
    ('retr_deadline_extend', 'PUYO_LIVE_24H_RETR_DEADLINE_EXTEND', 'PUYO_RETR_DEADLINE_EXTEND'),
)


def validate(value: bool, retr: bool, key: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{key}はboolです')
    if value and not retr:
        raise ValueError(f'{key}にはlive_24h_retrが必要です')
    return value


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name, _, _ in FLAGS:
        parser.add_argument('--live-24h-' + name.replace('_', '-'),
                            action=argparse.BooleanOptionalAction, default=False)


def retr_v2_values(data: dict, retr: bool) -> dict:
    return {f'live_24h_{name}': validate(data.get(f'live_24h_{name}', False), retr,
                                      f'live_24h_{name}') for name, _, _ in FLAGS}


def retr_v2_settings(active: bool) -> dict:
    retr = os.environ.get('PUYO_LIVE_24H_RETR') == '1'
    return {name: active and validate(os.environ.get(env) == '1', retr, f'live_24h_{name}')
            for name, env, _ in FLAGS}


def configure_retr_v2(options: Any, parser: Any, active: bool, retr: bool) -> None:
    try:
        values = retr_v2_values(vars(options), retr)
    except ValueError as exc:
        parser.error(str(exc))
    for name, env, _ in FLAGS:
        os.environ[env] = str(int(active and values[f'live_24h_{name}']))


def configure_child(settings: dict) -> None:
    for name, _, env in FLAGS:
        value = validate(settings.get(name, False), settings.get('retr', False), f'live_24h_{name}')
        os.environ[env] = str(int(value))
