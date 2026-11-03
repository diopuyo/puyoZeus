"""置きによる連鎖分割の起動設定。既定は OFF。"""
from __future__ import annotations

import os
from typing import Any

ENV_CHAIN_SPLIT_PLACE = 'PUYO_CHAIN_SPLIT_PLACE'


def validate_split(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_chain_split_placeはboolです')
    return value


def split_enabled() -> bool:
    return os.environ.get(ENV_CHAIN_SPLIT_PLACE) == '1'


def set_split_env(value: bool) -> None:
    """OFF は外部から残った値も消し、子の設定を優先する。"""
    if validate_split(value):
        os.environ[ENV_CHAIN_SPLIT_PLACE] = '1'
    else:
        os.environ.pop(ENV_CHAIN_SPLIT_PLACE, None)


def configure_split(value: bool, active: bool, parser: Any) -> None:
    try:
        set_split_env(validate_split(value) and active)
    except ValueError as exc:
        parser.error(str(exc))
