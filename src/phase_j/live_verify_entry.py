"""入口だけ画面確認を短い間隔で再試行する。合否の条件は変更しない。"""
from __future__ import annotations

import os

from src.production_config import LIVE_VERIFY_ENTRY_FAST_RETRY_DEFAULT

FLAG_ENV = 'PUYO_LIVE_VERIFY_ENTRY_FAST_RETRY'
ENTRY_VERIFY_PERIOD_SEC = 0.25


def enabled() -> bool:
    """試作の旗は既定 OFF。起動時の明示値だけを読む。"""
    return os.environ.get(FLAG_ENV, str(int(LIVE_VERIFY_ENTRY_FAST_RETRY_DEFAULT))) == '1'


def verification_period(verified: bool, regular_period: float, fast_retry: bool) -> float:
    """検証済みと旗 OFF は既存の間隔をそのまま返す。"""
    return ENTRY_VERIFY_PERIOD_SEC if fast_retry and not verified else regular_period
