"""画像検査の例外を出力へ残し、次の画像から検査を再開する。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
import time
import traceback
from typing import Any, Callable

OUTPUT_ENV = 'PUYO_LIVE_SCREEN_ERROR_OUTPUT'
LOG_FILE = 'screen_errors.jsonl'
_LOCK = Lock()


def configure_output(output: Path | None) -> None:
    """記録先は起動ごとに更新し、認識の子プロセスにも引き継ぐ。"""
    os.environ.pop(OUTPUT_ENV, None)
    if output is not None:
        os.environ[OUTPUT_ENV] = str(Path(output).resolve())


def record_error(stage: str) -> None:
    """画像自体を保存せず、発生箇所と traceback を記録する。"""
    logger = logging.getLogger(__name__)
    logger.exception('画像検査で例外が発生しました: %s。次の画像で再試行します', stage)
    output = os.environ.get(OUTPUT_ENV)
    if not output:
        return
    row = dict(event='screen_error', stage=stage, at=time.time(), pid=os.getpid(),
               traceback=traceback.format_exc())
    try:
        with _LOCK, (Path(output) / LOG_FILE).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError:
        logger.exception('画像検査の例外を保存できませんでした')


def check_screen(stage: str, check: Callable[..., bool], *args: Any) -> bool:
    """例外の画像は不合格とし、入力ループや worker を終了させない。"""
    try:
        return check(*args)
    except Exception:
        record_error(stage)
        return False
