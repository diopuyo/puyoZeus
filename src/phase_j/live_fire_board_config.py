"""B3の撃った盤を明示ONのときだけ使う配信設定。"""
from __future__ import annotations

import argparse
import os
from typing import Any

from src.production_config import LIVE_B3_FIRE_BOARD_DEFAULT

ENV_ENABLED = 'PUYO_LIVE_B3_FIRE_BOARD'


def enabled() -> bool:
    """認識・評価workerが同じ旗を読む。未設定は既定OFF。"""
    return os.environ.get(ENV_ENABLED, str(int(LIVE_B3_FIRE_BOARD_DEFAULT))) == '1'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--live-b3-fire-board', action=argparse.BooleanOptionalAction,
                        default=LIVE_B3_FIRE_BOARD_DEFAULT)


def user_value(data: dict, error_type: type[Exception] = ValueError) -> bool:
    value = data.get('live_b3_fire_board', LIVE_B3_FIRE_BOARD_DEFAULT)
    if type(value) is not bool:
        raise error_type('live_b3_fire_boardはboolです')
    return value


def configure(options: Any, parser: argparse.ArgumentParser) -> None:
    try:
        value = user_value(vars(options))
    except ValueError as error:
        parser.error(str(error))
    os.environ[ENV_ENABLED] = str(int(value))
